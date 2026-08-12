"""
STEP 1 -- Freeze the current state before any corrections.

Run from the repo root, notebook CLOSED:
    python freeze_and_manifest.py

Creates ../aim-ahead-asthma-FROZEN-<date>/ containing a copy of the notebooks,
outputs, and key artifacts, plus canonical_run_manifest.md with SHA-256 hashes,
package versions, and a record of what this run produced.

This is read-only with respect to your working repo: nothing is modified.
"""
import hashlib, json, os, platform, shutil, subprocess, sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent
STAMP = datetime.now().strftime("%Y%m%d_%H%M")
FROZEN = REPO.parent / f"aim-ahead-asthma-FROZEN-{STAMP}"

# What to preserve (data/ is large; parquet files are listed by hash, not copied)
COPY = ["notebooks", "outputs", "README.md", "requirements.txt"]
HASH_ONLY = ["data/processed"]

def sha256(p, chunk=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()

def walk_files(root):
    for dp, dn, fn in os.walk(root):
        dn[:] = [d for d in dn if d not in {".venv", "__pycache__", ".ipynb_checkpoints", ".git"}]
        for f in fn:
            yield Path(dp) / f

print(f"Freezing to: {FROZEN}")
FROZEN.mkdir(parents=True, exist_ok=True)

hashes = []
for item in COPY:
    src = REPO / item
    if not src.exists():
        continue
    dst = FROZEN / item
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__", ".ipynb_checkpoints"))
        for f in walk_files(src):
            hashes.append((str(f.relative_to(REPO)), f.stat().st_size, sha256(f)))
    else:
        shutil.copy2(src, dst)
        hashes.append((item, src.stat().st_size, sha256(src)))

for item in HASH_ONLY:
    src = REPO / item
    if src.exists():
        for f in walk_files(src):
            hashes.append((str(f.relative_to(REPO)), f.stat().st_size, sha256(f)))

# environment
try:
    pkgs = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                          capture_output=True, text=True, timeout=120).stdout
except Exception as e:
    pkgs = f"(pip freeze failed: {e})"

# key results, if present
metrics = ""
mj = REPO / "outputs" / "model_comparison_metrics.json"
if mj.exists():
    metrics = json.dumps(json.load(open(mj)), indent=2)[:4000]

lines = [
    "# Canonical run manifest",
    "",
    f"- Frozen at: {datetime.now().isoformat(timespec='seconds')}",
    f"- Snapshot: `{FROZEN.name}`",
    f"- Python: {sys.version.split()[0]} on {platform.platform()}",
    "",
    "## Purpose",
    "",
    "Preserves the state of the analysis that reproduces the submitted Model A",
    "point estimates (AUC 0.827, sensitivity 0.780, specificity 0.720) prior to",
    "the R3 corrections. Do not modify this snapshot.",
    "",
    "## File hashes (SHA-256)",
    "",
    "| File | Bytes | SHA-256 |",
    "|---|---:|---|",
]
for rel, size, h in sorted(hashes):
    lines.append(f"| `{rel}` | {size} | `{h}` |")

lines += ["", "## model_comparison_metrics.json (excerpt)", "", "```json", metrics, "```",
          "", "## Installed packages", "", "```", pkgs.strip(), "```", ""]

(FROZEN / "canonical_run_manifest.md").write_text("\n".join(lines), encoding="utf-8")
print(f"  files hashed: {len(hashes)}")
print(f"  manifest: {FROZEN / 'canonical_run_manifest.md'}")
print("\nFREEZE COMPLETE. Safe to begin corrections on the working repo.")
