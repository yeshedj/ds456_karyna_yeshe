#!/usr/bin/env python3
"""
evaluate_wer.py

Compare reference transcripts (from chat_to_whisper.py, *_cinderella.txt)
against Whisper transcripts (from run_whisper.py, *_whisper.txt) and report
word error rate (WER) and its parts. Standard library only.

Usage:
    python evaluate_wer.py --ref cleaned/ --hyp whisper_out/
    python evaluate_wer.py --ref cleaned/ --hyp whisper_out/ --csv results.csv
    python evaluate_wer.py --ref cleaned/ --hyp whisper_out/ --show-alignment

Files are paired by name: kansas03_cinderella.txt <-> kansas03_whisper.txt

Metrics:
    WER = (substitutions + deletions + insertions) / words in reference
    Overall WER pools all errors over all words (long transcripts count more).
"""

import argparse
import csv
import re
from collections import Counter
from pathlib import Path


def normalize(text: str):
    """Lowercase, drop punctuation, split hyphens, return a list of words."""
    t = text.lower().replace("’", "'").replace("‘", "'")
    t = re.sub(r"[-–—_]", " ", t)
    t = re.sub(r"[^a-z0-9' ]", " ", t)
    t = re.sub(r"(?<![a-z0-9])'|'(?![a-z0-9])", " ", t)  # stray quote marks
    return t.split()


def align(ref, hyp):
    """Word-level Levenshtein alignment -> list of (op, ref_word, hyp_word).
    op: C correct, S substitution, D deletion (missed), I insertion (extra)."""
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            d[i][j] = min(d[i - 1][j - 1] + cost, d[i - 1][j] + 1, d[i][j - 1] + 1)

    ops, i, j = [], n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            ops.append(("C" if ref[i - 1] == hyp[j - 1] else "S", ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            ops.append(("D", ref[i - 1], None))
            i -= 1
        else:
            ops.append(("I", None, hyp[j - 1]))
            j -= 1
    ops.reverse()
    return ops


def score(ops):
    c = Counter(op for op, _, _ in ops)
    s, d, i, correct = c["S"], c["D"], c["I"], c["C"]
    n = s + d + correct
    return {"ref_words": n, "correct": correct, "substitutions": s,
            "deletions": d, "insertions": i,
            "wer": (s + d + i) / n if n else float("nan")}


def show_alignment(ops):
    out = []
    for op, r, h in ops:
        if op == "C":
            out.append(r)
        elif op == "S":
            out.append(f"[{r}->{h}]")
        elif op == "D":
            out.append(f"[-{r}]")
        else:
            out.append(f"[+{h}]")
    return " ".join(out)


def key_of(path: Path):
    return re.sub(r"_(cinderella|whisper)$", "", path.stem)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", required=True, help="folder of reference .txt files")
    ap.add_argument("--hyp", required=True, help="folder of Whisper .txt files")
    ap.add_argument("--csv", help="save per-file results to this CSV")
    ap.add_argument("--show-alignment", action="store_true",
                    help="print each transcript with errors marked inline")
    args = ap.parse_args()

    refs = {key_of(p): p for p in Path(args.ref).glob("*.txt")}
    hyps = {key_of(p): p for p in Path(args.hyp).glob("*.txt")}
    shared = sorted(set(refs) & set(hyps))
    for k in sorted(set(refs) ^ set(hyps)):
        side = "reference" if k in refs else "whisper"
        print(f"  note: '{k}' only has a {side} file, skipped")
    if not shared:
        raise SystemExit("No matching file pairs found.")

    rows, confusions = [], Counter()
    totals = Counter()
    for k in shared:
        ref = normalize(refs[k].read_text(encoding="utf-8"))
        hyp = normalize(hyps[k].read_text(encoding="utf-8"))
        ops = align(ref, hyp)
        res = score(ops)
        res["file"] = k
        rows.append(res)
        for key in ("ref_words", "correct", "substitutions", "deletions", "insertions"):
            totals[key] += res[key]
        confusions.update((r, h) for op, r, h in ops if op == "S")

        if args.show_alignment:
            print(f"\n=== {k} ===\n{show_alignment(ops)}")

    print(f"\n{'file':<24}{'ref words':>10}{'S':>5}{'D':>5}{'I':>5}{'WER':>9}")
    for r in rows:
        print(f"{r['file']:<24}{r['ref_words']:>10}{r['substitutions']:>5}"
              f"{r['deletions']:>5}{r['insertions']:>5}{r['wer']:>9.1%}")

    overall = ((totals["substitutions"] + totals["deletions"] + totals["insertions"])
               / totals["ref_words"])
    mean = sum(r["wer"] for r in rows) / len(rows)
    print(f"\nFiles compared:      {len(rows)}")
    print(f"Overall WER (pooled): {overall:.1%}")
    print(f"Mean per-file WER:    {mean:.1%}")
    print(f"Substitutions {totals['substitutions']}, deletions {totals['deletions']}, "
          f"insertions {totals['insertions']} over {totals['ref_words']} reference words")

    if confusions:
        print("\nMost common substitutions (reference -> whisper):")
        for (r, h), n in confusions.most_common(10):
            print(f"  {r} -> {h}  x{n}")

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["file", "ref_words", "correct",
                                              "substitutions", "deletions",
                                              "insertions", "wer"])
            w.writeheader()
            w.writerows(rows)
        print(f"\nSaved {args.csv}")


if __name__ == "__main__":
    main()
