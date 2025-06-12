#!/usr/bin/env python
"""
fix_enhanced_mail_call_file.py
──────────────────────────────
One-click cleaner for the long “enhanced_mail_call_analysis.py” file that was
scrambled by smart quotes / fancy dashes / copy-paste artefacts.

► Edit the two variables below if your files are named differently.
► Run:  python fix_enhanced_mail_call_file.py
"""

# ────────────────────────────── CONFIG ──────────────────────────────
IN_FILE  = "enhanced_mail_call_analysis.py"        # original messy file
OUT_FILE = "enhanced_mail_call_analysis_CLEAN.py"  # cleaned copy to write
# ────────────────────────────────────────────────────────────────────

from pathlib import Path
import re, sys

# map of weird → ascii
REPL = {
    "“": '"', "”": '"', "‘": "'", "’": "'",
    "—": "-", "–": "-", "‒": "-", "‐": "-", "−": "-",
    "…": "...",
}
SMART = re.compile("|".join(map(re.escape, REPL)))

DUnder_MAIN_PAT = re.compile(
    r"if\s+[\*\_]*name[\*\_]*\s*==\s*[\"']{1,2}__main__[\"']{1,2}\s*:",
    flags=re.IGNORECASE,
)
LOGGER_PAT = re.compile(
    r"logging\.getLogger\(\s*\*\*name\*\*\s*\)", flags=re.IGNORECASE
)

def clean_text(txt: str) -> str:
    txt = "".join(ch for ch in txt if ch.isprintable() or ch in "\r\n\t")
    txt = SMART.sub(lambda m: REPL[m.group(0)], txt)
    txt = LOGGER_PAT.sub("logging.getLogger(__name__)", txt)
    txt = DUnder_MAIN_PAT.sub('if __name__ == "__main__":', txt)

    # Ensure plain shebang on first line
    if not txt.startswith("#!/usr/bin/env python"):
        txt = "#!/usr/bin/env python\n" + txt.lstrip()
    return txt

def main() -> None:
    here = Path(__file__).resolve().parent
    src  = here / IN_FILE
    dst  = here / OUT_FILE

    if not src.exists():
        sys.exit(f"❌  Could not find {src}")

    raw = src.read_text(encoding="utf-8", errors="ignore")
    fixed = clean_text(raw)
    dst.write_text(fixed, encoding="utf-8")
    print(f"✔ Clean version written → {dst.relative_to(here)}")

if __name__ == "__main__":
    main()