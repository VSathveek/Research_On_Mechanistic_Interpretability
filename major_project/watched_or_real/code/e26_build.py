# E26 v2 builder. usage: build.py <min|llama> <arm> <layers win|all> <pos last|span> <lam> <seed> <smoke 0|1> <owner>
# arm: none real refusal pcs8 pcs32 rand8 rand32 full (objective inv) | act (act_paper) | bct | natural | sysprompt
import sys, io, base64, json, os, numpy as np, nbformat
HERE = os.path.dirname(os.path.abspath(__file__))
RES = "D:/Research_On_Mechanistic_Interpretability/major_project/watched_or_real/results"
CFG = {"min": ("mistralai/Ministral-8B-Instruct-2410", 15, "S2_directions_and_steering_ministral", False),
       "llama": ("meta-llama/Llama-3.1-8B-Instruct", 10, "R_llama31_8b_replication", True)}
mdl, arm, layers, pos, lam, seed, smoke, owner = sys.argv[1:9]; smoke = smoke == "1"; seed = int(seed); lam = float(lam)
name, sel, rdir, gated = CFG[mdl]
if gated: assert owner == "varanasilvsssathveek", "gated-model notebooks carry the HF token: user's own account only"
s = open(f"{HERE}/e26.py", encoding="utf-8").read()
def rep(a, b):
    global s
    assert s.count(a) == 1, a[:60]; s = s.replace(a, b)
rep('MODEL, SEL = "mistralai/Ministral-8B-Instruct-2410", 15', f'MODEL, SEL = "{name}", {sel}')
dR = np.load(f"{RES}/{rdir}/dR.npy"); lw = list(range(sel - 2, sel + 3))
b1 = io.BytesIO(); np.savez_compressed(b1, layers=np.array(lw), dR=dR[lw].astype(np.float32)); rep('BLOB = ""', f'BLOB = "{base64.b64encode(b1.getvalue()).decode()}"')
b2 = io.BytesIO(); np.savez_compressed(b2, dR=dR.astype(np.float32)); rep('DRALL64 = ""', f'DRALL64 = "{base64.b64encode(b2.getvalue()).decode()}"')
if gated: rep('HF_TOKEN = ""', f'HF_TOKEN = "{open(f"{HERE}/../hf_token.txt").read().strip()}"')
method = {"act": "act_paper", "bct": "bct", "natural": "natural", "sysprompt": "sysprompt"}.get(arm, "inv")
rep('METHOD = "inv"', f'METHOD = "{method}"')
rep('ARM, LAYERS, POS, LAM, SEED = "refusal", "all", "span", 1.0, 0', f'ARM, LAYERS, POS, LAM, SEED = "{arm}", "{layers}", "{pos}", {lam}, {seed}')
if not smoke: rep("SMOKE = True", "SMOKE = False")
compile(s, "e26", "exec")
tag = f"{mdl}-{arm}-{layers}-{pos}-l{lam:g}-s{seed}".replace(".", "p") + ("-smoke" if smoke else "")
out = f"{HERE}/v2_{tag}"; os.makedirs(out, exist_ok=True)
nb = nbformat.v4.new_notebook(); nb.cells = [nbformat.v4.new_code_cell(s)]
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}; nb.metadata["language_info"] = {"name": "python"}
nbformat.write(nb, open(f"{out}/e26.ipynb", "w", encoding="utf-8"))
slug = f"{owner}/wor-e26v2-{tag}"
json.dump({"id": slug, "title": slug.split("/")[1].replace("-", " "), "code_file": "e26.ipynb", "language": "python", "kernel_type": "notebook",
           "is_private": "true", "enable_gpu": "true", "enable_tpu": "false", "enable_internet": "true", "machine_shape": "",
           "dataset_sources": [f"{owner}/wor-e26-bases-{mdl}"], "competition_sources": [], "kernel_sources": [], "model_sources": []},
          open(f"{out}/kernel-metadata.json", "w"), indent=2)
print("built", slug, out)
