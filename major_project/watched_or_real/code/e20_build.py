# E20 builder: E19 notebook re-targeted to Llama-3.1-8B (replication) or re-run on Ministral with a different training seed.
# usage: build.py <model: llama|min> <method> <seed> <smoke 0|1>
import sys, io, base64, json, os, re, numpy as np, nbformat
OLD = "C:/Users/ASUS/AppData/Local/Temp/claude/d--project-opensource/f9695e91-7cd5-41cb-8ae0-afdfa8729af8/scratchpad"
NEW = "C:/Users/ASUS/AppData/Local/Temp/claude/2dbff466580ab3d5/f9695e91-7cd5-41cb-8ae0-afdfa8729af8/scratchpad"
RES = "D:/Research_On_Mechanistic_Interpretability/major_project/watched_or_real/results"
mdl, method, seed, smoke = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4] == "1"
s = open(f"{OLD}/e19/e19.py", encoding="utf-8").read()
def rep(a, b):
    global s
    assert a in s, a[:60]; s = s.replace(a, b, 1)
if mdl == "llama":
    dR = np.load(f"{RES}/R_llama31_8b_replication/dR.npy"); layers = [8, 9, 10, 11, 12]
    rep('MODEL, SEL = "mistralai/Ministral-8B-Instruct-2410", 15', 'MODEL, SEL = "meta-llama/Llama-3.1-8B-Instruct", 10')
    tok = open(f"{NEW}/hf_token.txt").read().strip()
    rep("from datasets import load_dataset\n", f'from datasets import load_dataset\nfrom huggingface_hub import login\nHF_TOKEN = "{tok}"\nif HF_TOKEN: login(HF_TOKEN)\n')
else:
    dR = np.load(f"{RES}/S2_directions_and_steering_ministral/dR.npy"); layers = [13, 14, 15, 16, 17]
rep("torch.manual_seed(0); random.seed(0); np.random.seed(0)", f"SEED = {seed}\ntorch.manual_seed(SEED); random.seed(SEED); np.random.seed(SEED)")
rep("random.Random(ep).shuffle(EX)", "random.Random(ep + 1000 * SEED).shuffle(EX)")
buf = io.BytesIO(); np.savez_compressed(buf, layers=np.array(layers), dR=dR[layers].astype(np.float32))
rep('BLOB = ""', f'BLOB = "{base64.b64encode(buf.getvalue()).decode()}"')
rep('METHOD = "bsct"', f'METHOD = "{method}"')
if not smoke: rep("SMOKE = True", "SMOKE = False")
compile(s, "e20", "exec")
tag = f"{mdl}-{method.replace('_', '-')}-s{seed}" + ("-smoke" if smoke else "")
owner = "varanasilvsssathveek" if mdl == "llama" else "vanamalaaravind"   # HF token only ever goes to the user's own account
out = f"{NEW}/e20/b_{tag}"; os.makedirs(out, exist_ok=True)
nb = nbformat.v4.new_notebook(); nb.cells = [nbformat.v4.new_code_cell(s)]
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}; nb.metadata["language_info"] = {"name": "python"}
nbformat.write(nb, open(f"{out}/e20.ipynb", "w", encoding="utf-8"))
slug = f"{owner}/watched-or-real-e20-{tag}"
json.dump({"id": slug, "title": slug.split("/")[1].replace("-", " "), "code_file": "e20.ipynb", "language": "python", "kernel_type": "notebook",
           "is_private": "true", "enable_gpu": "true", "enable_tpu": "false", "enable_internet": "true", "machine_shape": "",
           "dataset_sources": [], "competition_sources": [], "kernel_sources": [], "model_sources": []}, open(f"{out}/kernel-metadata.json", "w"), indent=2)
print("built", slug, out)
