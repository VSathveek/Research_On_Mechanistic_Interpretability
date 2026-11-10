# E26 run queue. usage: queue.py <min|llama> <owner>[=kaggle_config_dir] [<owner>[=dir] ...]
# Deviation D2: several accounts per model; each run remembers its account.
# Keeps <= 2 sessions on the account, pushes runs in priority order, downloads outputs, retries a failed run once.
import sys, os, subprocess, time, json
HERE = os.path.dirname(os.path.abspath(__file__))
mdl = sys.argv[1]; OWN = [a.split("=", 1) + ([] if "=" in a else [None]) for a in sys.argv[2:]]
ENV = {}
for o, c in OWN:
    e = dict(os.environ)
    if c: e["KAGGLE_CONFIG_DIR"] = c
    ENV[o] = e
OWNERS = [o for o, _ in OWN]
OUTROOT = f"{HERE}/../o26/full/{mdl}"; os.makedirs(OUTROOT, exist_ok=True); STATE = f"{OUTROOT}/_state.json"
R = []
R += [("natural", "all", "span", 1, 0), ("sysprompt", "all", "span", 1, 0)]
for s in [0, 1, 2]:
    R += [(a, "all", "span", 1, s) for a in ["refusal", "real", "pcs8", "rand8", "full", "act", "bct"]]
R += [("none", "all", "span", 0, 0)]
R += [(a, l, p, 1, 0) for a in ["refusal", "pcs8"] for l, p in [("win", "last"), ("all", "last"), ("win", "span")]]
R += [(a, "all", "span", x, 0) for a in ["refusal", "pcs8"] for x in [4, 16]] + [("act", "all", "span", x, 0) for x in [0.3, 3]]
R += [(a, "all", "span", 1, 0) for a in ["pcs32", "rand32"]]
def tag(r): a, l, p, lam, s = r; return f"{mdl}-{a}-{l}-{p}-l{float(lam):g}-s{s}".replace(".", "p")
def sh(cmd, owner, **k):
    try: return subprocess.run(cmd, capture_output=True, text=True, env=ENV[owner], **k)
    except Exception as e: return subprocess.CompletedProcess(cmd, 1, "", f"sh error: {e}")
st = json.load(open(STATE)) if os.path.exists(STATE) else {}
def save(): json.dump(st, open(STATE, "w"), indent=1)
def own(t): return st[t].get("owner", OWNERS[0])
def status(t):
    o = sh(["kaggle", "kernels", "status", f"{own(t)}/wor-e26v2-{t}"], own(t)).stdout
    for x in ["COMPLETE", "ERROR", "CANCEL", "RUNNING", "QUEUED"]:
        if x in o: return x
    return "UNKNOWN"
print(f"{len(R)} runs for {mdl} on {OWNERS}", flush=True)
while True:
    running = [t for t, v in st.items() if v["state"] == "pushed"]
    for t in running:
        s_ = status(t)
        if s_ == "COMPLETE":
            d = f"{OUTROOT}/{t}"; os.makedirs(d, exist_ok=True)
            dl = sh(["kaggle", "kernels", "output", f"{own(t)}/wor-e26v2-{t}", "-p", d], own(t), timeout=3600)
            if dl.returncode and not any(f.startswith("e26_result_") for f in os.listdir(d)): print(time.strftime("%H:%M"), t, "download retry", dl.stderr[-150:], flush=True); continue
            ok = any(f.startswith("e26_result_") for f in os.listdir(d)); st[t]["state"] = "done" if ok else "failed"
            print(time.strftime("%H:%M"), t, st[t]["state"], flush=True); save()
        elif s_ in ("ERROR", "CANCEL"):
            st[t]["state"] = "failed"; print(time.strftime("%H:%M"), t, "ERROR", flush=True); save()
    for t, v in list(st.items()):
        if v["state"] == "failed" and v.get("tries", 1) < 2: v["state"] = "todo"; v["tries"] = v.get("tries", 1) + 1; v["version"] = True
    running = [t for t, v in st.items() if v["state"] == "pushed"]
    todo = [r for r in R if st.get(tag(r), {"state": "todo"})["state"] == "todo"]
    if not todo and not running: break
    for owner in OWNERS:
        n_own = sum(1 for t in running if own(t) == owner)
        while todo and n_own < 2:
            r = todo[0]; t = tag(r); a, l, p, lam, s = r
            b = sh([sys.executable, "-P", f"{HERE}/build.py", mdl, a, l, p, str(lam), str(s), "0", owner], owner)
            if b.returncode: print("build failed", t, owner, b.stderr[-500:], flush=True); st[t] = {"state": "buildfail"}; save(); todo.pop(0); continue
            o = sh(["kaggle", "kernels", "push", "-p", f"{HERE}/v2_{t}"], owner)
            if "successfully pushed" in o.stdout:
                st[t] = {"state": "pushed", "owner": owner, "tries": st.get(t, {}).get("tries", 1), "at": time.strftime("%m-%d %H:%M")}; running.append(t); todo.pop(0); n_own += 1
                print(time.strftime("%H:%M"), "pushed", t, owner, flush=True); save()
            else:
                print(time.strftime("%H:%M"), "push refused", t, owner, (o.stdout + o.stderr)[-120:].strip().replace(chr(10), " "), flush=True); break
    time.sleep(300)
print("ALL DONE", {k: v["state"] for k, v in st.items()}, flush=True)
