import json, base64, sys, nbformat, os
HERE = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(HERE, "..", "am_prompts", "blackmail_explicit-america_replacement")
d = {"system": open(os.path.join(P, "system_prompt.txt"), encoding="utf-8").read(),
     "user": open(os.path.join(P, "user_prompt.txt"), encoding="utf-8").read(),
     "email": open(os.path.join(P, "email_content.txt"), encoding="utf-8").read()}
b64 = base64.b64encode(json.dumps(d).encode("utf-8")).decode("ascii")
body = open(os.path.join(HERE, "pilot_body.py"), encoding="utf-8").read()
smoke = "--full" not in sys.argv
if not smoke:
    assert "SMOKE = True" in body; body = body.replace("SMOKE = True", "SMOKE = False")
src = f'PROMPTS_B64 = "{b64}"\n' + body
compile(src, "pilot", "exec")
nb = nbformat.v4.new_notebook(); nb.cells = [nbformat.v4.new_code_cell(src)]
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nb.metadata["language_info"] = {"name": "python"}
os.makedirs(os.path.join(HERE, "kernel"), exist_ok=True)
nbformat.write(nb, open(os.path.join(HERE, "kernel", "wr.ipynb"), "w", encoding="utf-8"))
json.dump({"id": "vanamalaaravind/watched-or-real-pilot", "title": "watched or real pilot", "code_file": "wr.ipynb",
           "language": "python", "kernel_type": "notebook", "is_private": "true", "enable_gpu": "true",
           "enable_tpu": "false", "enable_internet": "true", "machine_shape": "", "dataset_sources": [],
           "competition_sources": [], "kernel_sources": [], "model_sources": []},
          open(os.path.join(HERE, "kernel", "kernel-metadata.json"), "w"), indent=2)
print("built", "SMOKE" if smoke else "FULL", len(src), "chars; hf token present:", "hf_" in src)
