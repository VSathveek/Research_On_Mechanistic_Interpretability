# Builds e24.py from e21.py: BSCT that protects a direction DISCOVERED in E23 (bsct_dir) or "real" + that direction (bsct_combo).
import sys
SRC, OUTF = sys.argv[1], sys.argv[2]
s = open(SRC, encoding="utf-8").read()
def rep(a, b):
    global s
    assert s.count(a) == 1, (s.count(a), a[:70]); s = s.replace(a, b)

rep(s[:s.index("import os;")], """# E24 -- Protect the belief each model actually uses. BSCT whose consistency loss sits on the E23-discovered framing direction
# (Llama: role-play pc2; Ministral: framing pc0) or on the supervised residual compliance direction, alone (bsct_dir) or added to
# the original "real" loss (bsct_combo). Same data, LoRA, schedule and evaluation as E19-E21. Primary evaluation on JBB goals 50-99,
# which E23 never used to select or orient any direction.
""")
rep('METHOD = "bsct"', 'METHOD = "bsct_dir"\nDIRB64 = ""')
rep('if METHOD in ("bsct", "bsct_sub", "act", "act_paper"):', '''if METHOD in ("bsct_dir", "bsct_combo"):
    _d = np.load(io.BytesIO(base64.b64decode(DIRB64))); DL = int(_d["layer"]); DU = torch.tensor(_d["u"].astype(np.float32)); DGAP = float(_d["gap"])
    assert DL in LW; print(f"protected direction at layer {DL}, gap {DGAP:.3f}")
if METHOD in ("bsct", "bsct_sub", "act", "act_paper", "bsct_dir", "bsct_combo"):''')
rep("""                        if METHOD == "bsct":
                            r = RV[l].to(h.device); term = (((h - tgt) @ r) / GAP[l]) ** 2""",
"""                        if METHOD in ("bsct", "bsct_dir", "bsct_combo"):
                            term = 0.0
                            if METHOD != "bsct_dir": r = RV[l].to(h.device); term = (((h - tgt) @ r) / GAP[l]) ** 2
                            if METHOD != "bsct" and l == DL: term = term + len(LW) * (((h - tgt) @ DU.to(h.device)) / DGAP) ** 2""")
s = s.replace("e21_", "e24_").replace("\\nE21 |", "\\nE24 |")
open(OUTF, "w", encoding="utf-8").write(s); compile(s, "e24", "exec"); print("ok", len(s))
