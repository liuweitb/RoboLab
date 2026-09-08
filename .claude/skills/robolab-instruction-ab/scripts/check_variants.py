#!/usr/bin/env python3
"""Verify every task in a directory defines the instruction keys the A/B will compare.

A task whose dict lacks a key silently falls back to `default`, which would compare a condition
against itself and quietly halve the experiment. Run this BEFORE launching.

Parses the task files as text, so it needs no Isaac Sim and runs in a second.
"""
import argparse, ast, glob, os, sys

ap = argparse.ArgumentParser()
ap.add_argument("--tasks-dir", required=True, help="directory holding task .py files")
ap.add_argument("--conditions", nargs="+", required=True, help="instruction keys the A/B compares")
ap.add_argument("--quiet-ok", action="store_true", help="only print tasks that are missing a key")
A = ap.parse_args()


def task_info(path):
    """(class name, instruction dict) for the Task subclass in this file, or None."""
    tree = ast.parse(open(path).read())
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        bases = [b.id for b in node.bases if isinstance(b, ast.Name)]
        if "Task" not in bases:
            continue
        for stmt in node.body:
            targets = stmt.targets if isinstance(stmt, ast.Assign) else \
                      ([stmt.target] if isinstance(stmt, ast.AnnAssign) else [])
            for t in targets:
                if isinstance(t, ast.Name) and t.id == "instruction" and stmt.value is not None:
                    try:
                        return node.name, ast.literal_eval(stmt.value)
                    except Exception:
                        return node.name, None
        return node.name, None
    return None


files = sorted(f for f in glob.glob(os.path.join(A.tasks_dir, "*.py"))
               if os.path.basename(f) != "__init__.py")
ok, bad, unparsed = [], [], []
for f in files:
    info = task_info(f)
    if not info:
        continue
    cls, instr = info
    if instr is None:
        unparsed.append((cls, os.path.basename(f))); continue
    missing = [c for c in A.conditions if c not in instr]
    (bad if missing else ok).append((cls, os.path.basename(f), missing, sorted(instr)))

if not A.quiet_ok:
    for cls, fn, _, keys in ok:
        print(f"  OK       {cls:34s} {fn:38s} keys: {', '.join(keys)}")
for cls, fn, missing, keys in bad:
    print(f"  MISSING  {cls:34s} {fn:38s} lacks: {', '.join(missing)}  has: {', '.join(keys)}")
for cls, fn in unparsed:
    print(f"  UNPARSED {cls:34s} {fn:38s} instruction is not a literal dict; check by hand")

print(f"\n{len(ok)} task(s) define all of {A.conditions}; {len(bad)} missing a key; "
      f"{len(unparsed)} unparsed.")
if bad:
    print("Exclude those tasks from the A/B or add the missing variant. A missing key falls back "
          "to `default`, which would compare a condition against itself.")
sys.exit(1 if bad else 0)
