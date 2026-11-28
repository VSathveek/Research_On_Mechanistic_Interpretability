"""
Compile the paper on Kaggle (CPU-only kernel), since no TeX distribution is installed locally.

Embeds main.tex, references.bib, generated/* and figures/*.pdf into a single script kernel
that installs TeX Live, runs latexmk, and leaves main_study.pdf and main_study.log in the output.

Usage:
    python compile_on_kaggle.py            # build + push
    python compile_on_kaggle.py --no-push  # build only
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SLUG = "cgsae-study-compile"
USER = "varanasilvsssathveek"
KDIR = os.path.join(HERE, "..", "kernels", SLUG)


def payload() -> dict[str, str]:
    files = {"main_study.tex": "main_study.tex", "references.bib": "references.bib"}
    for sub in ("generated", "figures"):
        for f in sorted(os.listdir(os.path.join(HERE, sub))):
            if sub == "figures" and not f.endswith(".pdf"):
                continue
            if sub == "generated" and not f.endswith(".tex"):
                continue
            files[f"{sub}/{f}"] = f"{sub}/{f}"
    out = {}
    for rel in files:
        with open(os.path.join(HERE, rel), "rb") as fh:
            out[rel] = base64.b64encode(fh.read()).decode()
    return out


TEMPLATE = r'''
import base64, os, subprocess, sys, shutil
FILES = __FILES__
W = "/kaggle/working/paper"
os.makedirs(W, exist_ok=True)
for rel, b in FILES.items():
    p = os.path.join(W, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "wb").write(base64.b64decode(b))
print("wrote", len(FILES), "files", flush=True)

pk = ["texlive-latex-base", "texlive-latex-recommended", "texlive-latex-extra",
      "texlive-fonts-recommended", "texlive-science", "texlive-pictures", "texlive-publishers", "lmodern", "latexmk"]
subprocess.run("apt-get update -qq", shell=True)
r = subprocess.run("DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends "
                   + " ".join(pk), shell=True)
print("apt exit", r.returncode, flush=True)

r = subprocess.run("latexmk -pdf -interaction=nonstopmode -halt-on-error main_study.tex",
                   shell=True, cwd=W, capture_output=True, text=True)
print("latexmk exit", r.returncode, flush=True)
log = open(os.path.join(W, "main_study.log"), errors="replace").read() if os.path.exists(os.path.join(W, "main_study.log")) else ""
for line in log.splitlines():
    if line.startswith("!") or "Warning" in line and ("undefined" in line or "Citation" in line or "Reference" in line):
        print("LOG:", line, flush=True)
print("---- log tail ----")
print(log[-3000:], flush=True)
for f in ("main_study.pdf", "main_study.log", "main_study.blg"):
    if os.path.exists(os.path.join(W, f)):
        shutil.copy(os.path.join(W, f), "/kaggle/working/" + f)
print("pdf present:", os.path.exists("/kaggle/working/main_study.pdf"), flush=True)
'''


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-push", action="store_true")
    a = ap.parse_args()
    os.makedirs(KDIR, exist_ok=True)
    code = TEMPLATE.replace("__FILES__", json.dumps(payload()))
    with open(os.path.join(KDIR, f"{SLUG}.py"), "w", encoding="utf-8", newline="\n") as f:
        f.write(code)
    meta = {"id": f"{USER}/{SLUG}", "title": "CGSAE study compile", "code_file": f"{SLUG}.py",
            "language": "python", "kernel_type": "script", "is_private": True,
            "enable_gpu": False, "enable_internet": True, "dataset_sources": [],
            "competition_sources": [], "kernel_sources": []}
    json.dump(meta, open(os.path.join(KDIR, "kernel-metadata.json"), "w"), indent=2)
    print(f"built {SLUG} ({os.path.getsize(os.path.join(KDIR, SLUG + '.py')) / 1024:.0f} KB)")
    if a.no_push:
        return
    r = subprocess.run([sys.executable, "-m", "kaggle", "kernels", "push", "-p", KDIR],
                       capture_output=True, text=True)
    print((r.stdout or r.stderr).strip())


if __name__ == "__main__":
    main()
