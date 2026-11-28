"""Fail loudly if main.tex uses a number-macro that numbers.tex does not define, or if an
\\input'd table is missing. Cheap guard against a compile that silently drops a number."""

import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
tex = open(os.path.join(HERE, "main.tex"), encoding="utf-8").read()
nums = open(os.path.join(HERE, "generated", "numbers.tex"), encoding="utf-8").read()

defined = set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", nums))
local = set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", tex))
used = set(re.findall(r"\\([a-z]+[A-Z][A-Za-z]*)", tex))      # camelCase = our macros
missing = sorted(u for u in used if u not in defined and u not in local)

inputs = re.findall(r"\\input\{([^}]+)\}", tex)
absent = [i for i in inputs if not os.path.exists(os.path.join(HERE, i))]
figs = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex)
nofig = [f for f in figs if not os.path.exists(os.path.join(HERE, f))]

print(f"number macros used: {len(used)} | defined: {len(defined)}")
print(f"missing macros : {missing or 'none'}")
print(f"missing inputs : {absent or 'none'}")
print(f"missing figures: {nofig or 'none'}")
sys.exit(1 if (missing or absent or nofig) else 0)
