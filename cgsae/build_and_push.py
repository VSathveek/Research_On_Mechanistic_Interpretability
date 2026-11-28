"""
Assemble the cgsae/src modules into one self-contained Kaggle script kernel and push it.

Kaggle script kernels take a single code file and no CLI arguments, so this concatenates
the four modules (stripping their internal cross-imports) and bakes the run arguments into
a sys.argv assignment at the bottom.

Kernel naming
-------------
    slug   cgsae-v2-<axis>-<model>-l<layer>[-<suffix>]
    title  CGSAE v2 <axis> <model> L<layer>

so every run is identifiable from the Kaggle listing alone, e.g.

    cgsae-v2-smoke-gpt2-l8
    cgsae-v2-sentiment-gpt2-l8
    cgsae-v2-toxicity-gpt2-l8
    cgsae-v2-sentiment-qwen1p5b-l14

Usage
-----
    python build_and_push.py --axis sentiment --model gpt2 --layer 8
    python build_and_push.py --axis sentiment --model gpt2 --layer 8 --smoke --no-push
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "src")
KERNELS = os.path.join(HERE, "kernels")
MODULES = ["cgsae_core.py", "cgsae_model.py", "cgsae_data.py", "run_experiment.py"]
INTERNAL = {"cgsae_core", "cgsae_model", "cgsae_data", "run_experiment"}

KAGGLE_USER = "varanasilvsssathveek"

PREAMBLE = '''# ===========================================================================
# CG-SAE v2 - auto-generated Kaggle kernel. Do not edit here.
# Source of truth: cgsae/src/*.py ; regenerate with cgsae/build_and_push.py
# ===========================================================================
from __future__ import annotations

import subprocess
import sys


def _pip(*pkgs):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=False)


# Kaggle images drift; pin the pieces this run actually depends on.
_pip("transformers==4.44.2", "datasets==2.20.0", "scikit-learn", "scipy", "pandas")
print("environment ready", flush=True)

import torch  # noqa: E402
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "", flush=True)
'''

MODEL_SHORT = {
    "gpt2": "gpt2",
    "gpt2-medium": "gpt2m",
    "gpt2-large": "gpt2l",
    "Qwen/Qwen2.5-1.5B-Instruct": "qwen1p5b",
    "Qwen/Qwen2.5-0.5B-Instruct": "qwen0p5b",
    "EleutherAI/pythia-410m": "pythia410m",
}


# ---------------------------------------------------------------------------
def strip_internal_imports(source: str, drop_future: bool, drop_main: bool) -> str:
    """Remove cross-module imports (and optionally __future__ / the __main__ block).

    Uses the AST rather than regex so multi-line parenthesised imports are handled.
    """
    tree = ast.parse(source)
    lines = source.splitlines()
    drop: set[int] = set()

    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            if node.module in INTERNAL or (node.module == "__future__" and drop_future):
                drop.update(range(node.lineno, (node.end_lineno or node.lineno) + 1))
        elif isinstance(node, ast.Import):
            if any(a.name in INTERNAL for a in node.names):
                drop.update(range(node.lineno, (node.end_lineno or node.lineno) + 1))
        elif drop_main and isinstance(node, ast.If):
            # if __name__ == "__main__": ...
            test = node.test
            if (isinstance(test, ast.Compare)
                    and isinstance(test.left, ast.Name) and test.left.id == "__name__"):
                drop.update(range(node.lineno, (node.end_lineno or node.lineno) + 1))

    return "\n".join(l for i, l in enumerate(lines, start=1) if i not in drop)


def build_script(run_args: list[str]) -> str:
    parts = [PREAMBLE]
    for i, mod in enumerate(MODULES):
        src = open(os.path.join(SRC, mod), encoding="utf-8").read()
        body = strip_internal_imports(
            src, drop_future=True, drop_main=(mod == "run_experiment.py"))
        parts.append(f"\n\n# {'=' * 74}\n# module: {mod}\n# {'=' * 74}\n{body}")

    argv = ", ".join(json.dumps(a) for a in run_args)
    parts.append(f'''

# ============================================================================
# entry point - arguments baked in at build time
# ============================================================================
if __name__ == "__main__":
    sys.argv = ["run_experiment.py", {argv}]
    print("ARGS:", sys.argv[1:], flush=True)
    main()
''')
    return "".join(parts)


# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--axis", default="sentiment",
                   choices=["sentiment", "toxicity", "refusal"])
    p.add_argument("--model", default="gpt2")
    p.add_argument("--layer", type=int, default=8)
    p.add_argument("--suffix", default="", help="extra tag appended to the kernel slug")
    p.add_argument("--job", default="main", choices=["main", "exp1"],
                   help="main = run_experiment.py (GPU); exp1 = exp1_synthetic.py (CPU only)")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--no-push", action="store_true")
    p.add_argument("--public", action="store_true")
    p.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                   help="everything after --extra is passed straight to run_experiment.py")
    a = p.parse_args()
    a.extra = sanitise_extra(a.extra)
    return a


# Flags that belong to this builder and must never reach the kernel. `--extra` is a
# REMAINDER, so anything typed after it - including builder flags - is captured verbatim;
# a stray `--no-push` once got baked into a kernel's argv and the run died on
# "unrecognized arguments".
BUILDER_ONLY = {"--no-push", "--public", "--extra"}
BUILDER_ONLY_WITH_VALUE = {"--job", "--suffix"}


def sanitise_extra(extra: list[str]) -> list[str]:
    out, skip = [], False
    for tok in extra:
        if skip:
            skip = False
            continue
        if tok in BUILDER_ONLY:
            print(f"  note: dropping builder-only flag {tok} from kernel args")
            continue
        if tok in BUILDER_ONLY_WITH_VALUE:
            print(f"  note: dropping builder-only flag {tok} from kernel args")
            skip = True
            continue
        out.append(tok)
    return out


def build_exp1_script(run_args: list[str]) -> str:
    """The synthetic experiment is self-contained - no module concatenation needed."""
    src = open(os.path.join(SRC, "exp1_synthetic.py"), encoding="utf-8").read()
    body = strip_internal_imports(src, drop_future=True, drop_main=True)
    argv = ", ".join(json.dumps(x) for x in run_args)
    preamble = (
        "# Auto-generated Kaggle kernel. Source: cgsae/src/exp1_synthetic.py\n"
        "from __future__ import annotations\n\n"
        "import subprocess, sys\n"
        "subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'scipy', "
        "'pandas', 'numpy'], check=False)\n"
        "print('environment ready', flush=True)\n"
    )
    return (preamble + "\n\n" + body + f'''

if __name__ == "__main__":
    sys.argv = ["exp1_synthetic.py", {argv}]
    print("ARGS:", sys.argv[1:], flush=True)
    main()
''')


def main() -> None:
    a = parse_args()
    short = MODEL_SHORT.get(a.model, a.model.split("/")[-1].lower().replace(".", "p"))

    if a.job == "exp1":
        slug = "cgsae-v2-exp1-synthetic" + (f"-{a.suffix}" if a.suffix else "")
        title = "CGSAE v2 exp1 synthetic" + (f" {a.suffix}" if a.suffix else "")
        run_args = ["--out", "/kaggle/working/results"] + \
                   a.extra
        kdir = os.path.join(KERNELS, slug)
        os.makedirs(kdir, exist_ok=True)
        code_file = f"{slug}.py"
        path = os.path.join(kdir, code_file)
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(build_exp1_script(run_args))
        meta = {
            "id": f"{KAGGLE_USER}/{slug}", "title": title, "code_file": code_file,
            "language": "python", "kernel_type": "script", "is_private": not a.public,
            "enable_gpu": False,          # pure CPU - do not spend GPU quota on it
            "enable_internet": True,
            "dataset_sources": [], "competition_sources": [], "kernel_sources": [],
        }
        with open(os.path.join(kdir, "kernel-metadata.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        compile(open(path, encoding="utf-8").read(), path, "exec")
        print(f"built {path} ({os.path.getsize(path)/1024:.1f} KB)  [CPU kernel]")
        print(f"  kernel id : {meta['id']}")
        print(f"  args      : {' '.join(run_args)}")
        if a.no_push:
            print("  (skipped push)")
            return
        r = subprocess.run([sys.executable, "-m", "kaggle", "kernels", "push", "-p", kdir],
                           capture_output=True, text=True)
        print(r.stdout.strip() or r.stderr.strip())
        return

    if a.smoke:
        slug = f"cgsae-v2-smoke-{short}-l{a.layer}"
        title = f"CGSAE v2 smoke {short} L{a.layer}"
    else:
        slug = f"cgsae-v2-{a.axis}-{short}-l{a.layer}"
        title = f"CGSAE v2 {a.axis} {short} L{a.layer}"
    if a.suffix:
        slug += f"-{a.suffix}"
        title += f" {a.suffix}"

    run_args = ["--axis", a.axis, "--model", a.model, "--layer", str(a.layer),
                "--out", "/kaggle/working/results"]
    if a.smoke:
        run_args.append("--smoke")
    run_args += a.extra

    kdir = os.path.join(KERNELS, slug)
    os.makedirs(kdir, exist_ok=True)
    code_file = f"{slug}.py"
    with open(os.path.join(kdir, code_file), "w", encoding="utf-8", newline="\n") as f:
        f.write(build_script(run_args))

    meta = {
        "id": f"{KAGGLE_USER}/{slug}",
        "title": title,
        "code_file": code_file,
        "language": "python",
        "kernel_type": "script",
        "is_private": not a.public,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
    }
    with open(os.path.join(kdir, "kernel-metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # Compile check before spending GPU quota on a syntax error.
    path = os.path.join(kdir, code_file)
    compile(open(path, encoding="utf-8").read(), path, "exec")
    size = os.path.getsize(path)
    print(f"built {path} ({size/1024:.1f} KB)")
    print(f"  kernel id : {meta['id']}")
    print(f"  args      : {' '.join(run_args)}")

    if a.no_push:
        print("  (skipped push)")
        return

    r = subprocess.run([sys.executable, "-m", "kaggle", "kernels", "push", "-p", kdir],
                       capture_output=True, text=True)
    print(r.stdout.strip() or r.stderr.strip())
    if r.returncode == 0:
        print(f"  watch: https://www.kaggle.com/code/{meta['id'].split('/')[0]}/{slug}")


if __name__ == "__main__":
    main()
