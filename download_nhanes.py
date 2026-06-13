"""
Download NHANES files for AIM-AHEAD pediatric asthma analysis.
Cycles: 2007-2008 (E), 2009-2010 (F), 2011-2012 (G)
Files saved to: data/raw/
"""

from pathlib import Path
import urllib.request
import urllib.error

FILE_STEMS = [
    "DEMO", "BMX", "MCQ", "SPX", "COTNAL", "FSQ", "HIQ", "HSQ",
    "HUQ", "PFQ", "ECQ", "RDQ", "SMQ", "SMQFAM", "SMQRTU"
]

# Cycle suffix -> first year of cycle (used in URL path)
CYCLES = {
    "E": "2007",
    "F": "2009",
    "G": "2011",
}

OUT_DIR = Path("data/raw")
OUT_DIR.mkdir(parents=True, exist_ok=True)

base_url = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public"
total = len(FILE_STEMS) * len(CYCLES)
done = skipped = failed = 0

def is_valid_xpt(path: Path) -> bool:
    """SAS transport files start with 'HEADER RECORD'. HTML error pages don't."""
    try:
        with open(path, "rb") as f:
            head = f.read(20)
        return head.startswith(b"HEADER RECORD")
    except Exception:
        return False

for suffix, year in CYCLES.items():
    for stem in FILE_STEMS:
        filename = f"{stem}_{suffix}.xpt"
        local_path = OUT_DIR / filename

        if local_path.exists() and is_valid_xpt(local_path):
            print(f"  [skip] {filename} (already valid)")
            skipped += 1
            continue

        url = f"{base_url}/{year}/DataFiles/{stem}_{suffix}.xpt"
        try:
            urllib.request.urlretrieve(url, local_path)
            if not is_valid_xpt(local_path):
                local_path.unlink(missing_ok=True)
                print(f"  [FAIL] {filename}  (server returned non-XPT content)")
                failed += 1
                continue
            size_kb = local_path.stat().st_size // 1024
            print(f"  [ ok ] {filename}  ({size_kb} KB)")
            done += 1
        except urllib.error.HTTPError as e:
            print(f"  [FAIL] {filename}  HTTP {e.code}")
            failed += 1
        except Exception as e:
            print(f"  [FAIL] {filename}  {e}")
            failed += 1

print()
print(f"Summary: {done} downloaded, {skipped} skipped, {failed} failed (of {total} total)")