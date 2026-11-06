#!/usr/bin/env python3
import argparse
import csv
import re
from collections import Counter
from pathlib import Path


def normalize(text: str):
    """Clean and split text into normalized words for transcript comparison."""
    t = text.lower().replace("’", "'").replace("‘", "'")
    t = re.sub(r"[-–—_]", " ", t)
    t = re.sub(r"[^a-z0-9' ]", " ", t)
    t = re.sub(r"(?<![a-z0-9])'|'(?![a-z0-9])", " ", t)
    return t.split()


def align_words(ref, hyp):
    """Levenshtein alignment returning (op, ref_word, hyp_word).
    Operations: C = Correct, S = Substitution, D = Deletion, I = Insertion
    """
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j - 1] + cost,
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
            )

    ops, i, j = [], n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            op = "C" if ref[i - 1] == hyp[j - 1] else "S"
            ops.append((op, ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append(("D", ref[i - 1], None))
            i -= 1
        else:
            ops.append(("I", None, hyp[j - 1]))
            j -= 1

    ops.reverse()
    return ops


def calculate_wer(ops):
    counts = Counter(op for op, _, _ in ops)
    subs, dels, ins, correct = counts["S"], counts["D"], counts["I"], counts["C"]
    n_ref = subs + dels + correct

    return {
        "ref_words": n_ref,
        "correct": correct,
        "substitutions": subs,
        "deletions": dels,
        "insertions": ins,
        "wer": (subs + dels + ins) / n_ref if n_ref > 0 else float("nan"),
    }


def format_inline_alignment(ops):
    tokens = []
    for op, r, h in ops:
        if op == "C":
            tokens.append(r)
        elif op == "S":
            tokens.append(f"[{r}->{h}]")
        elif op == "D":
            tokens.append(f"[-{r}]")
        else:
            tokens.append(f"[+{h}]")
    return " ".join(tokens)


def extract_file_key(path: Path):
    return re.sub(r"_(cinderella|whisper)$", "", path.stem)


def run():
    parser = argparse.ArgumentParser(description="Evaluate Word Error Rate (WER) between transcripts.")
    parser.add_argument("--ref", required=True, help="Directory containing reference transcript .txt files")
    parser.add_argument("--hyp", required=True, help="Directory containing hypothesis Whisper .txt files")
    parser.add_argument("--csv", help="Path to save summary CSV metrics")
    parser.add_argument("--errors-csv", help="Path to save word-level alignment details CSV")
    parser.add_argument("--show-alignment", action="store_true", help="Print inline alignments to console")

    args = parser.parse_args()

    ref_map = {extract_file_key(p): p for p in Path(args.ref).glob("*.txt")}
    hyp_map = {extract_file_key(p): p for p in Path(args.hyp).glob("*.txt")}
    common_keys = sorted(set(ref_map) & set(hyp_map))

    if not common_keys:
        raise SystemExit("Error: No matching transcript pairs found.")

    results, confusions, all_alignments = [], Counter(), []
    totals = Counter()

    for key in common_keys:
        ref_words = normalize(ref_map[key].read_text(encoding="utf-8"))
        hyp_words = normalize(hyp_map[key].read_text(encoding="utf-8"))

        if not ref_words:
            print(f"Skipping '{key}': reference file is empty.")
            continue

        ops = align_words(ref_words, hyp_words)
        stats = calculate_wer(ops)
        stats["file"] = key

        results.append(stats)
        all_alignments.append((key, len(ref_words), ops))

        for metric in ("ref_words", "correct", "substitutions", "deletions", "insertions"):
            totals[metric] += stats[metric]

        confusions.update((r, h) for op, r, h in ops if op == "S")

        if args.show_alignment:
            print(f"\n=== {key} ===")
            print(format_inline_alignment(ops))

    if not results:
        raise SystemExit("Error: No non-empty reference files found.")

    print(f"\n{'File':<24}{'Ref Words':>10}{'S':>5}{'D':>5}{'I':>5}{'WER':>9}")
    print("-" * 58)
    for r in results:
        print(f"{r['file']:<24}{r['ref_words']:>10}{r['substitutions']:>5}"
              f"{r['deletions']:>5}{r['insertions']:>5}{r['wer']:>9.1%}")

    total_errors = totals["substitutions"] + totals["deletions"] + totals["insertions"]
    overall_wer = total_errors / totals["ref_words"]
    mean_wer = sum(r["wer"] for r in results) / len(results)

    print("\nSummary:")
    print(f"  Files compared:      {len(results)}")
    print(f"  Overall WER (pooled): {overall_wer:.1%}")
    print(f"  Mean per-file WER:    {mean_wer:.1%}")
    print(f"  Total errors: {totals['substitutions']} S, {totals['deletions']} D, {totals['insertions']} I "
          f"across {totals['ref_words']} reference words.")

    if confusions:
        print("\nTop 10 Substitutions (Ref -> Hyp):")
        for (ref_w, hyp_w), count in confusions.most_common(10):
            print(f"  {ref_w} -> {hyp_w} (x{count})")

    if args.errors_csv:
        with open(args.errors_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["file", "op", "ref_word", "hyp_word", "rel_pos"])
            for key, total_ref, ops in all_alignments:
                pos = 0
                for op, r, h in ops:
                    if op != "I":
                        pos += 1
                    writer.writerow([key, op, r or "", h or "", round(pos / total_ref, 4)])
        print(f"\nSaved word-level alignment log to {args.errors_csv}")

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["file", "ref_words", "correct", "substitutions", "deletions", "insertions", "wer"]
            )
            writer.writeheader()
            writer.writerows(results)
        print(f"Saved summary metrics to {args.csv}")


if __name__ == "__main__":
    run()