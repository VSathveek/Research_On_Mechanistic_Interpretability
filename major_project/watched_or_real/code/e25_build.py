# E25 builder. usage: build.py <min|llama> <parts> <smoke 0|1>
import sys, io, base64, json, os, numpy as np, pandas as pd, nbformat
HERE = os.path.dirname(os.path.abspath(__file__))
RES = "D:/Research_On_Mechanistic_Interpretability/major_project/watched_or_real/results"
CFG = {"min":   ("mistralai/Ministral-8B-Instruct-2410", 15, "S2_directions_and_steering_ministral", False, "pc0", "up", "vanamalaaravind"),
       "llama": ("meta-llama/Llama-3.1-8B-Instruct", 10, "R_llama31_8b_replication", True, "pc2", "down", "varanasilvsssathveek")}
mdl, parts, smoke = sys.argv[1], sys.argv[2], sys.argv[3] == "1"
name, sel, rdir, gated, dname, kind, owner = CFG[mdl]
s = open(f"{HERE}/e25.py", encoding="utf-8").read()
def rep(a, b):
    global s
    assert s.count(a) == 1, a[:60]; s = s.replace(a, b)
rep('MODEL, SEL = "mistralai/Ministral-8B-Instruct-2410", 15', f'MODEL, SEL = "{name}", {sel}')
dR = np.load(f"{RES}/{rdir}/dR.npy"); layers = list(range(sel - 2, sel + 3))
buf = io.BytesIO(); np.savez_compressed(buf, layers=np.array(layers), dR=dR[layers].astype(np.float32))
rep('BLOB = ""', f'BLOB = "{base64.b64encode(buf.getvalue()).decode()}"')
if gated: rep('HF_TOKEN = ""', f'HF_TOKEN = "{open(f"{HERE}/../hf_token.txt").read().strip()}"')
E23 = f"{RES}/E23_beyond_the_atlas/{mdl}__"
Zd = np.load(E23 + "e23_dirs.npz"); T = pd.read_csv(E23 + "e23_directions.csv").set_index("name")
mc, mr = float(T.loc[dname, "m_comp"]), float(T.loc[dname, "m_ref"])
b2 = io.BytesIO(); np.savez_compressed(b2, u=Zd[dname].astype(np.float32), layer=Zd["layer"], gap=np.array(mc - mr)); rep('DIRB64 = ""', f'DIRB64 = "{base64.b64encode(b2.getvalue()).decode()}"')
b3 = io.BytesIO(); np.savez_compressed(b3, atlas=Zd["atlas"].astype(np.float32), meta=np.array(json.dumps(dict(kind=kind, m_comp=mc, m_ref=mr, name=dname))))
rep('E23B64 = ""', f'E23B64 = "{base64.b64encode(b3.getvalue()).decode()}"')
rep('PARTS = "ABC"', f'PARTS = "{parts}"')
AD = {"bsct": f"{mdl}_bsct", "act_paper": f"{mdl}_act_paper", "dir": f"{mdl}_dir_{dname}"} if set(parts) & set("AB") else {}
rep('ADAPTERS = ""', f"ADAPTERS = '{json.dumps(AD)}'")
if smoke: rep("N_RAND = 19", "N_RAND = 2")
else: rep("SMOKE = True", "SMOKE = False")
compile(s, "e25", "exec")
tag = f"{mdl}-{parts.lower()}" + ("-smoke" if smoke else "")
out = f"{HERE}/b_{tag}"; os.makedirs(out, exist_ok=True)
nb = nbformat.v4.new_notebook(); nb.cells = [nbformat.v4.new_code_cell(s)]
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}; nb.metadata["language_info"] = {"name": "python"}
nbformat.write(nb, open(f"{out}/e25.ipynb", "w", encoding="utf-8"))
slug = f"{owner}/watched-or-real-e25-{tag}"
json.dump({"id": slug, "title": slug.split("/")[1].replace("-", " "), "code_file": "e25.ipynb", "language": "python", "kernel_type": "notebook",
           "is_private": "true", "enable_gpu": "true", "enable_tpu": "false", "enable_internet": "true", "machine_shape": "",
           "dataset_sources": [f"{owner}/wor-e25-{mdl}-adapters"] if AD else [], "competition_sources": [], "kernel_sources": [], "model_sources": []},
          open(f"{out}/kernel-metadata.json", "w"), indent=2)
print("built", slug)
