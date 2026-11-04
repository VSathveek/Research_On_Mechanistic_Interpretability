# Builds e21.py from e19.py: any model (Gemma in fp32 compute, in-notebook "real" direction when no BLOB is given) and a
# METHOD = "probe" mode, the cheap causal test used to PREDICT which defence wins before any training.
import sys
OLD = sys.argv[1]; OUTF = sys.argv[2]
s = open(f"{OLD}/e19/e19.py", encoding="utf-8").read()
def rep(a, b):
    global s
    assert s.count(a) == 1, (s.count(a), a[:70]); s = s.replace(a, b)

rep(s[:s.index("import os;")], """# E21 -- Does a cheap causal test predict which defence wins? Same pipeline as E19 on new models (Qwen2.5-7B, Gemma-3-4B),
# plus METHOD = "probe": on held-out fiction jailbreaks, clamp the "real" reading back to explicit-real (b = 1) at the decision
# position vs moving a random direction by the same amount; on plain requests, clamp it to explicit-simulated (b = -1).
# Selective restore effect = ASR drop from restoring "real" minus ASR drop from the matched random move.
# The prediction rule (threshold on the selective restore effect, calibrated on Ministral and Llama) is committed BEFORE the
# Qwen/Gemma defence runs are read.
""")
rep('METHOD = "bsct"                         # natural | bsct | bsct_sub | act | act_paper',
    'METHOD = "bsct"                         # probe | natural | bsct | bsct_sub | act | act_paper\nHF_TOKEN = ""')
rep("""Z = np.load(io.BytesIO(base64.b64decode(BLOB)))
LW = [int(x) for x in Z["layers"]]; DR = {l: Z["dR"][k].astype(np.float32) for k, l in enumerate(LW)}
""", "")
rep("from datasets import load_dataset\n", "from datasets import load_dataset\nfrom huggingface_hub import login\n")
rep('tok = AutoTokenizer.from_pretrained(MODEL);', 'if HF_TOKEN: login(HF_TOKEN)' + chr(10) + 'tok = AutoTokenizer.from_pretrained(MODEL);')
rep("""model = AutoModelForCausalLM.from_pretrained(MODEL, device_map="auto", max_memory={0: "13GiB", 1: "13GiB"}, torch_dtype=torch.float16,
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)).eval()""",
"""
DT = torch.float32 if "gemma" in MODEL.lower() else torch.float16          # Gemma overflows in fp16
model = AutoModelForCausalLM.from_pretrained(MODEL, device_map="auto", max_memory={0: "13GiB", 1: "13GiB"}, torch_dtype=DT,
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=DT)).eval()
NL = getattr(model.config, "text_config", model.config).num_hidden_layers""")
# "real" direction: from the BLOB, or computed here with a rule fixed in advance
rep("""@torch.no_grad()
def last_states(texts, layers=LW, bs=8):""", """if BLOB:
    Z = np.load(io.BytesIO(base64.b64decode(BLOB)))
    LW = [int(x) for x in Z["layers"]]; DR = {l: Z["dR"][k].astype(np.float32) for k, l in enumerate(LW)}
else:   # same recipe as the original directions: explicit framing sentences on neutral tasks; layer = best held-out separation in the middle half
    @torch.no_grad()
    def _st(texts):
        out = []
        for i in range(0, len(texts), 8):
            enc = tok(texts[i:i + 8], return_tensors="pt", padding=True, add_special_tokens=False).to(EMB)
            out.append(torch.stack([h[:, -1].float().cpu() for h in model(**enc, output_hidden_states=True).hidden_states], 1))
        return torch.cat(out)                                                    # [N, L+1, D]
    TA, TB = alp[200:264], alp[264:328]
    P_, N_ = _st([chat(t, BASE + "\\n" + x) for x in RP[:2] for t in TA]), _st([chat(t, BASE + "\\n" + x) for x in RM[:2] for t in TA])
    Ph, Nh = _st([chat(t, BASE + "\\n" + RP[2]) for t in TB]), _st([chat(t, BASE + "\\n" + RM[2]) for t in TB])
    dall = (P_.mean(0) - N_.mean(0)).numpy(); sc = {}
    for l in range(round(0.25 * NL), round(0.75 * NL) + 1):
        u = torch.tensor(unit(dall[l])); a, b = Ph[:, l] @ u, Nh[:, l] @ u
        sc[l] = float((a.mean() - b.mean()) / torch.sqrt((a.var() + b.var()) / 2))
    SEL = max(sc, key=sc.get); LW = list(range(SEL - 2, SEL + 3)); DR = {l: dall[l] for l in LW}
    print(f"in-notebook real direction: layer {SEL} (held-out d' {sc[SEL]:.2f}); window {LW}")
    del P_, N_, Ph, Nh
@torch.no_grad()
def last_states(texts, layers=None, bs=8):
    layers = LW if layers is None else layers""")
rep("NLY = model.config.num_hidden_layers; ALL = list(range(1, NLY + 1))", "NLY = NL; ALL = list(range(1, NLY + 1))")
rep("""    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                                             layers_to_transform=list(range(0, NLY if METHOD == "act_paper" else max(LW))), task_type="CAUSAL_LM"))""",
"""    UP = NLY if METHOD == "act_paper" else max(LW)                              # text-decoder attention only (skips any vision tower)
    TM = [n for n, _ in model.named_modules() if re.search(r"layers\\.(\\d+)\\.self_attn\\.(q|k|v|o)_proj$", n) and "vision" not in n
          and int(re.search(r"layers\\.(\\d+)\\.", n).group(1)) < UP]
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=TM, task_type="CAUSAL_LM"))""")
rep('LAYMOD = next(m for n, m in model.named_modules() if isinstance(m, torch.nn.ModuleList) and n.endswith("layers") and len(m) > 20)',
    'LAYMOD = next(m for n, m in model.named_modules() if isinstance(m, torch.nn.ModuleList) and n.endswith("layers") and len(m) == NL and "vision" not in n)')
# probe mode
rep("""for s, items in [("jb", JB), ("xstest", [("safe", -1, q) for q in XS]), ("orbench", [("borderline", -1, q) for q in ORB]), ("creative", [("benign_fiction", -1, q) for q in CRE])]:
    outs = generate([chat(p) for _, _, p in items])""", """RND = {l: torch.tensor(unit(np.random.default_rng(900 + l).standard_normal(len(DR[l]))).astype(np.float32)) for l in LW}
def clamp_to(b, rnd):
    for l in LW:
        def hook(m, i, o, l=l):
            h = o[0] if isinstance(o, tuple) else o; h = h.clone(); x = h[:, -1]; d = RV[l].to(x.device, x.dtype)
            delta = (MS[l] + b * GAP[l]) - x @ d; v = RND[l].to(x.device, x.dtype) if rnd else d
            h[:, -1] = x + delta.unsqueeze(-1) * v
            return (h,) + tuple(o[1:]) if isinstance(o, tuple) else h
        HOOKS.append(LAYMOD[l - 1].register_forward_hook(hook))
FJ = [x for x in JB if x[0] in FIC_TEST]; BJ = [x for x in JB if x[0] == "bare"]
PLAN = ([("none", FJ + BJ, None, False), ("restore", FJ, 1.0, False), ("restore_rand", FJ, 1.0, True), ("sim", BJ, -1.0, False), ("sim_rand", BJ, -1.0, True)]
        if METHOD == "probe" else [("jb", JB, None, False), ("xstest", [("safe", -1, q) for q in XS], None, False),
                                   ("orbench", [("borderline", -1, q) for q in ORB], None, False), ("creative", [("benign_fiction", -1, q) for q in CRE], None, False)])
for s, items, cb, crnd in PLAN:
    if cb is not None: clamp_to(cb, crnd)
    outs = generate([chat(p) for _, _, p in items]); clear()
    if METHOD == "probe": items = [(f"{s}:{w}", gi, p) for w, gi, p in items]; s = 'jb'""")
rep('mm = load_dataset("cais/mmlu", "all", split="test").shuffle(seed=0).select(range(N_MMLU))',
    'mm = load_dataset("cais/mmlu", "all", split="test").shuffle(seed=0).select(range(0 if METHOD == "probe" else N_MMLU))')
rep('MMLU = corr / len(mm); print', 'MMLU = corr / max(len(mm), 1); print')
rep('judge("ibm-granite/granite-3.3-8b-instruct", [g for g in GEN if g["set"] != "jb"],', 'if METHOD != "probe": judge("ibm-granite/granite-3.3-8b-instruct", [g for g in GEN if g["set"] != "jb"],')
rep("J = D[D.set == \"jb\"]\n", """J = D[D.set == "jb"]
if METHOD == "probe":
    A = lambda c, ws: J[J.wrap.isin([f"{c}:{w}" for w in ws])].yes.mean() * 100
    pr = dict(model=MODEL, layer=SEL, fic_none=A("none", FIC_TEST), fic_restore=A("restore", FIC_TEST), fic_restore_rand=A("restore_rand", FIC_TEST),
              bare_none=A("none", ["bare"]), bare_sim=A("sim", ["bare"]), bare_sim_rand=A("sim_rand", ["bare"]),
              real_reading_fiction=float(read_fic.mean()), real_reading_bare=float(read_bare.mean()))
    pr["selective_restore"] = (pr["fic_none"] - pr["fic_restore"]) - (pr["fic_none"] - pr["fic_restore_rand"])
    pr["selective_sufficiency"] = (pr["bare_sim"] - pr["bare_none"]) - (pr["bare_sim_rand"] - pr["bare_none"])
    G = J.copy(); G["c"] = G.wrap.str.split(":").str[0]; G = G[G.wrap.str.split(":").str[1].isin(FIC_TEST)]
    pv = G.pivot_table(index=["goal", G.wrap.str.split(":").str[1]], columns="c", values="yes", aggfunc="first").dropna().astype(int)
    rng = np.random.default_rng(0); bs = []
    for _ in range(4000):
        i = rng.integers(0, len(pv), len(pv)); q = pv.iloc[i]; bs.append(100 * (q.restore_rand.mean() - q.restore.mean()))
    pr["selective_restore_ci"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    print("\\n" + "=" * 80 + f"\\nE21 PROBE | {MODEL} | layer {SEL}\\n" + "=" * 80)
    for k, v in pr.items(): print(f"   {k:>24}: {v}")
    json.dump(pr, open(f"{OUT}/e21_probe.json", "w")); print("DONE", f"{time.time()-t0:.0f}s")
""")
rep("for l in ALL], 1).half().cpu())", "for l in ALL], 1).to(DT).cpu())   # fp16 overflows on Gemma's large residual norms")
NLc = chr(10)
i = s.index("res = dict(method=METHOD"); s = s[:i] + 'if METHOD != "probe":' + NLc + NLc.join("    " + x for x in s[i:].split(NLc)) + NLc
s = s.replace("e19_", "e21_").replace("\\nE19 |", "\\nE21 |")
open(OUTF, "w", encoding="utf-8").write(s); compile(s, "e21", "exec"); print("ok", len(s))
