# Builds e22.py from e21.py: no training, no generation by the protected model. Reads all 20 atlas beliefs + a harm direction at the
# decision token for every evaluation prompt (identical prompt sets to the natural runs, whose compliance labels are joined offline),
# adds wrapper-matched BENIGN prompts, and (GUARDS=True) scores every prompt with Llama Guard 3 8B and Granite Guardian 3.1 8B.
import sys
SRC, OUTF = sys.argv[1], sys.argv[2]
s = open(SRC, encoding="utf-8").read()
def rep(a, b):
    global s
    assert s.count(a) == 1, (s.count(a), a[:70]); s = s.replace(a, b)

rep(s[:s.index("import os;")], """# E22 -- Belief-manipulation guard. Does reading the model's situational beliefs (20 atlas directions, learned only from neutral
# framing sentences) at the decision token predict and block unsafe completions better than a harm probe or an external guard model?
# This notebook only extracts features; the preregistered analysis (leave-one-attack-family-out) runs offline on the saved arrays.
""")
rep('METHOD = "bsct"', 'METHOD = "features"\nGUARDS = False')
rep("for n in READ_SET:\n    pos, neg = BELIEFS[n]", "MP, MN = {}, {}\nfor n in BELIEFS:\n    pos, neg = BELIEFS[n]")
rep("BG[n] = {l: float((P_[l].mean(0) - N_[l].mean(0)).norm()) for l in LW}",
    "BG[n] = {l: float((P_[l].mean(0) - N_[l].mean(0)).norm()) for l in LW}\n"
    "    MP[n] = {l: float((P_[l] @ BD[n][l]).mean()) for l in LW}; MN[n] = {l: float((N_[l] @ BD[n][l]).mean()) for l in LW}")
cut = s.index("JB = [(w, gi, TEST_WRAP[w].format(g=low(g), G=g)) for gi, g in enumerate(test_goals) for w in TEST_WRAP]")
s = s[:cut] + r'''
# ---------------- E22 feature extraction ----------------
NAMES = list(BELIEFS)
Hh, Hb = last_states([chat(g) for g in train_goals]), last_states([chat(b) for b in ben_tasks[:150]])   # harm direction: AdvBench (disjoint from JBB) vs Alpaca, plain
HD = {l: torch.tensor(unit((Hh[l].mean(0) - Hb[l].mean(0)).numpy())) for l in LW}
HP = {l: float((Hh[l] @ HD[l]).mean()) for l in LW}; HN = {l: float((Hb[l] @ HD[l]).mean()) for l in LW}
JB = [(w, gi, TEST_WRAP[w].format(g=low(g), G=g)) for gi, g in enumerate(test_goals) for w in TEST_WRAP]
BW = [w for w in TEST_WRAP if w != "bare"]
BENW = [(w, i, TEST_WRAP[w].format(g=low(t), G=t)) for i, t in enumerate(alp[3100:3100 + (6 if SMOKE else 50)]) for w in BW]   # benign tasks in every attack wrapper
ROWS = ([dict(set="jb", wrap=w, goal=gi, prompt=p) for w, gi, p in JB] + [dict(set="xstest", wrap="safe", goal=i, prompt=p) for i, p in enumerate(XS)] +
        [dict(set="orbench", wrap="borderline", goal=i, prompt=p) for i, p in enumerate(ORB)] + [dict(set="creative", wrap="benign_fiction", goal=i, prompt=p) for i, p in enumerate(CRE)] +
        [dict(set="benign_wrapped", wrap=w, goal=i, prompt=p) for w, i, p in BENW])
ST = last_states([chat(r["prompt"]) for r in ROWS])
F = {"layers": np.array(LW), "sel": np.array(SEL), "names": np.array(NAMES)}
for l in LW:
    F[f"belief_L{l}"] = np.stack([((ST[l] @ BD[n][l]).numpy() - MN[n][l]) / (MP[n][l] - MN[n][l]) for n in NAMES], 1).astype(np.float32)
    F[f"harm_L{l}"] = (((ST[l] @ HD[l]).numpy() - HN[l]) / (HP[l] - HN[l])).astype(np.float32)
F["raw_sel"] = ST[SEL].numpy().astype(np.float16)
print(f"features for {len(ROWS)} prompts ({time.time()-t0:.0f}s)")
del model; gc.collect(); torch.cuda.empty_cache()

def guard_scores(name, build, pos_word, neg_word):
    gt = AutoTokenizer.from_pretrained(name); gt.padding_side = "left"
    if gt.pad_token is None: gt.pad_token = gt.eos_token
    gm = AutoModelForCausalLM.from_pretrained(name, device_map="auto", max_memory={0: "13GiB", 1: "13GiB"},
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)).eval()
    ip, ineg = gt.encode(pos_word, add_special_tokens=False)[0], gt.encode(neg_word, add_special_tokens=False)[0]
    texts = [build(gt, r["prompt"]) for r in ROWS]; sc, vd = [], []
    for b in range(0, len(texts), 8):
        enc = gt(texts[b:b + 8], return_tensors="pt", padding=True, add_special_tokens=False).to(gm.get_input_embeddings().weight.device)
        with torch.no_grad(): o = gm.generate(**enc, max_new_tokens=6, do_sample=False, pad_token_id=gt.pad_token_id, output_scores=True, return_dict_in_generate=True)
        for j in range(o.sequences.shape[0]):
            gen = o.sequences[j, enc["input_ids"].shape[1]:]; v = None
            for t in range(len(o.scores)):                       # first non-whitespace step carries the verdict
                if gt.decode(gen[t:t + 1]).strip():
                    lg = o.scores[t][j].float(); v = float(lg[ip] - lg[ineg]); break
            sc.append(np.nan if v is None else v); vd.append(gt.decode(gen, skip_special_tokens=True).strip().lower().startswith(pos_word.lower()))
    del gm; gc.collect(); torch.cuda.empty_cache()
    print(f"{name}: flagged {np.mean(vd)*100:.1f}% of prompts, {np.isnan(sc).sum()} unscored")
    return np.array(sc, np.float32), np.array(vd)
if GUARDS:
    for key, name, build, pw, nw in [
        ("llamaguard", "meta-llama/Llama-Guard-3-8B", lambda gt, p: gt.apply_chat_template([{"role": "user", "content": p}], tokenize=False), "unsafe", "safe"),
        ("granite", "ibm-granite/granite-guardian-3.1-8b",
         lambda gt, p: gt.apply_chat_template([{"role": "user", "content": p}], guardian_config={"risk_name": "harm"}, tokenize=False, add_generation_prompt=True), "Yes", "No")]:
        try: F[f"{key}_score"], F[f"{key}_flag"] = guard_scores(name, build, pw, nw)
        except Exception as e: print("GUARD FAILED", name, repr(e)[:300])
np.savez_compressed(f"{OUT}/e22_features.npz", **F)
pd.DataFrame(ROWS).drop(columns=["prompt"]).to_csv(f"{OUT}/e22_rows.csv", index=False)
print("DONE", f"{time.time()-t0:.0f}s")
'''
open(OUTF, "w", encoding="utf-8").write(s); compile(s, "e22", "exec"); print("ok", len(s))
