"""Build + push Kaggle kernels for src/bucket_llm.py. Usage: python build_bucket_llm.py <slug> <args...>"""
import json, os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))
slug, args = sys.argv[1], sys.argv[2:]
src = open(os.path.join(HERE, "src", "bucket_llm.py"), encoding="utf-8").read()
src = src.replace('if __name__ == "__main__":\n    main()', "")
src = src.replace("from __future__ import annotations\n", "")
runs = [a.split() for a in " ".join(args).split(" ;; ")]
pre = ("import subprocess, sys\n"
       "subprocess.run([sys.executable,'-m','pip','install','-q','transformers==4.44.2','datasets==2.20.0'],check=False)\n")
body = pre + src + "\nfor argv in " + json.dumps(runs) + ":\n    sys.argv = ['bucket_llm.py'] + argv\n    print('ARGS', argv, flush=True)\n    main()\n"
kd = os.path.join(HERE, "kernels", slug); os.makedirs(kd, exist_ok=True)
open(os.path.join(kd, slug + ".py"), "w", encoding="utf-8", newline="\n").write(body)
compile(body, slug, "exec")
json.dump({"id": f"{os.environ.get("KAGGLE_USER","varanasilvsssathveek")}/{slug}", "title": slug.replace("-", " "), "code_file": slug + ".py",
           "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": True,
           "enable_internet": True, "dataset_sources": [], "competition_sources": [], "kernel_sources": []},
          open(os.path.join(kd, "kernel-metadata.json"), "w"), indent=2)
r = subprocess.run([sys.executable, "-m", "kaggle", "kernels", "push", "-p", kd], capture_output=True, text=True)
print(slug, "->", (r.stdout or r.stderr).strip()[:160])
