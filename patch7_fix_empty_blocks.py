"""
PATCH 7 -- repair empty blocks left by the commented-out lines.

Patches 5b/5c commented out statements such as bmi_zscore. Where such a line was
the ONLY body of an `if:` block, that leaves an empty block and Python raises
IndentationError. This inserts `pass` into any block whose body is now empty.

Run WITH THE NOTEBOOK CLOSED:
    python patch7_fix_empty_blocks.py

Backup -> notebooks/04_model.ipynb.bak7
"""
import json, shutil, os, re

HERE = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(HERE, "notebooks", "04_model.ipynb")
shutil.copy(PATH, PATH + ".bak7")
nb = json.load(open(PATH, encoding="utf-8"))

OPENER = re.compile(r"^(\s*)(if|elif|else|for|while|try|except|finally|with|def|class)\b.*:\s*$")

def indent_of(line):
    return len(line) - len(line.lstrip())

def fix(src):
    lines = src.splitlines()
    out, inserted = [], 0
    for i, line in enumerate(lines):
        out.append(line)
        m = OPENER.match(line)
        if not m:
            continue
        opener_indent = len(m.group(1))
        # find the next line that is real code (not blank, not a comment)
        body_found = False
        for nxt in lines[i + 1:]:
            if not nxt.strip():
                continue
            if nxt.lstrip().startswith("#"):
                continue
            body_found = indent_of(nxt) > opener_indent
            break
        if not body_found:
            out.append(" " * (opener_indent + 4) + "pass  # [R3 correction] body removed")
            inserted += 1
    return "\n".join(out) + ("\n" if src.endswith("\n") else ""), inserted

total, touched = 0, []
for i, cell in enumerate(nb["cells"]):
    if cell.get("cell_type") != "code":
        continue
    src = "".join(cell.get("source", []))
    new, n = fix(src)
    if n:
        cell["source"] = new.splitlines(keepends=True)
        touched.append(i)
        total += n

json.dump(nb, open(PATH, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(f"cells modified: {touched}")
print(f"empty blocks repaired: {total}")

# compile-check every code cell so we catch any remaining syntax error now,
# not three hours into a run
print("\nSyntax check:")
bad = 0
for i, cell in enumerate(nb["cells"]):
    if cell.get("cell_type") != "code":
        continue
    src = "".join(cell.get("source", []))
    if src.strip().startswith("!") or src.strip().startswith("%"):
        continue
    try:
        compile(src, f"cell{i}", "exec")
    except SyntaxError as e:
        bad += 1
        print(f"  !! cell {i}: {type(e).__name__}: {e.msg} (line {e.lineno})")
        if e.text:
            print(f"     {e.text.rstrip()}")
if bad == 0:
    print("  all code cells compile cleanly.")
print(f"\nBackup: {PATH}.bak7")
