# E24 builder. usage: build.py <min|llama|qwen|gemma> <method> <smoke 0|1>
import sys, io, base64, json, os, numpy as np, nbformat
HERE = os.path.dirname(os.path.abspath(__file__))
RES = "D:/Research_On_Mechanistic_Interpretability/major_project/watched_or_real/results"
CFG = {"min":   ("mistralai/Ministral-8B-Instruct-2410", 15, "S2_directions_and_steering_ministral", False),
       "llama": ("meta-llama/Llama-3.1-8B-Instruct", 10, "R_llama31_8b_replication", True),
       "gemma": ("google/gemma-3-4b-it", 30, "R_gemma3_4b_replication", True),
       "qwen":  ("Qwen/Qwen2.5-7B-Instruct", 0, None, False)}           # direction computed in the notebook
mdl, method, dirname, smoke = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4] == "1"   # usage: build.py <model> <bsct_dir|bsct_combo> <direction name> <smoke>
name, sel, rdir, gated = CFG[mdl]
s = open(f"{HERE}/e24.py", encoding="utf-8").read()
def rep(a, b):
    global s
    assert s.count(a) == 1, a[:60]; s = s.replace(a, b)
rep('MODEL, SEL = "mistralai/Ministral-8B-Instruct-2410", 15', f'MODEL, SEL = "{name}", {sel}')
if rdir:
    dR = np.load(f"{RES}/{rdir}/dR.npy"); layers = list(range(sel - 2, sel + 3))
    buf = io.BytesIO(); np.savez_compressed(buf, layers=np.array(layers), dR=dR[layers].astype(np.float32))
    rep('BLOB = ""', f'BLOB = "{base64.b64encode(buf.getvalue()).decode()}"')
if gated: rep('HF_TOKEN = ""', f'HF_TOKEN = "{open(f"{HERE}/../hf_token.txt").read().strip()}"')
rep('METHOD = "bsct_dir"', f'METHOD = "{method}"')
import pandas as pd
E23 = f"{RES}/E23_beyond_the_atlas/{mdl}__"
Zd = np.load(E23 + "e23_dirs.npz"); T = pd.read_csv(E23 + "e23_directions.csv").set_index("name")
buf2 = io.BytesIO(); np.savez_compressed(buf2, u=Zd[dirname].astype(np.float32), layer=Zd["layer"], gap=np.array(T.loc[dirname, "m_comp"] - T.loc[dirname, "m_ref"]))
rep('DIRB64 = ""', f'DIRB64 = "{base64.b64encode(buf2.getvalue()).decode()}"')
if not smoke: rep("SMOKE = True", "SMOKE = False")
compile(s, "e24", "exec")
owner = "varanasilvsssathveek" if gated else "vanamalaaravind"          # the HF token only ever goes to the user's own account
tag = f"{mdl}-{method.replace('_', '-')}-{dirname}" + ("-smoke" if smoke else "")
out = f"{HERE}/b_{tag}"; os.makedirs(out, exist_ok=True)
nb = nbformat.v4.new_notebook(); nb.cells = [nbformat.v4.new_code_cell(s)]
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}; nb.metadata["language_info"] = {"name": "python"}
nbformat.write(nb, open(f"{out}/e24.ipynb", "w", encoding="utf-8"))
slug = f"{owner}/watched-or-real-e24-{tag}"
json.dump({"id": slug, "title": slug.split("/")[1].replace("-", " "), "code_file": "e24.ipynb", "language": "python", "kernel_type": "notebook",
           "is_private": "true", "enable_gpu": "true", "enable_tpu": "false", "enable_internet": "true", "machine_shape": "",
           "dataset_sources": [], "competition_sources": [], "kernel_sources": [], "model_sources": []}, open(f"{out}/kernel-metadata.json", "w"), indent=2)
print("built", slug)
