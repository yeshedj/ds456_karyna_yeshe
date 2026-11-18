#!/usr/bin/env python3
"""
evaluate_wer.py  (extended)

Compare reference transcripts (*_cinderella.txt) with Whisper transcripts
(*_whisper.txt) using several complementary measures, because plain WER counts
harmless differences (a dropped stutter, "step sister" vs "stepsisters") as errors.

Measures
  Strict WER      word-level Levenshtein edit distance after basic normalization
                  (lowercase, no punctuation, hyphens -> spaces).
  Adjusted WER    the same, after two principled clean-ups applied to BOTH texts:
                  (a) collapse immediate repetitions ("the the", "in her old in her old")
                  (b) merge split/joined compounds ("step sisters" == "stepsisters")
  Soft WER        adjusted alignment where near-miss substitutions (aunts/aunt,
                  seemed/seems) cost a fraction of an error instead of a full one.
  Char edit dist. Levenshtein distance on characters -> CER and a 0-1 similarity.
  LCS             longest common subsequence of words: how much of the reference
                  survives, in order (precision / recall / F1, like ROUGE-L).
  Jaccard         overlap of the word sets (vocabulary) and of word bigrams
                  (local word order): |A and B| / |A or B|.
  POS             parts of speech (NLTK): error rates for content vs function words,
                  per-class breakdown, and how often a substitution keeps the POS.

Usage
    python evaluate_wer.py --ref cleaned --hyp whisper_out
    python evaluate_wer.py --ref cleaned --hyp whisper_out --csv results.csv
    python evaluate_wer.py --ref cleaned --hyp whisper_out --errors-csv errors.csv
    python evaluate_wer.py --ref cleaned --hyp whisper_out --show-alignment

Options
    --no-collapse-repeats / --no-merge-compounds   turn individual adjustments off
    --no-pos                                       skip POS tagging
    --errors-mode strict|adjusted                  which alignment feeds --errors-csv
                                                   (default adjusted)

POS tagging needs:  pip install nltk   (the tagger data downloads automatically)
Optional speed-up for character edit distance:  pip install rapidfuzz
"""

import argparse
import csv
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

# ----------------------------------------------------------------------------
# Tokenizing and normalization
# ----------------------------------------------------------------------------


def tokenize(text: str):
    """-> list of (norm, orig): norm is lowercase for matching, orig keeps the
    original case so the POS tagger sees "I", "Cinderella", etc."""
    t = text.replace("’", "'").replace("‘", "'")
    t = re.sub(r"[-–—_]", " ", t)
    t = re.sub(r"[^A-Za-z0-9' ]", " ", t)
    t = re.sub(r"(?<![A-Za-z0-9])'|'(?![A-Za-z0-9])", " ", t)   # stray quote marks
    return [(w.lower(), w) for w in t.split()]


def collapse_repeats(tokens, max_n=3):
    """Drop the first copy of an immediately repeated word or phrase
    (up to max_n words): 'the the girl' -> 'the girl'."""
    changed = True
    while changed:
        changed = False
        for n in range(max_n, 0, -1):
            ns = [t[0] for t in tokens]
            out, i = [], 0
            while i < len(tokens):
                if i + 2 * n <= len(tokens) and ns[i:i + n] == ns[i + n:i + 2 * n]:
                    i += n
                    changed = True
                else:
                    out.append(tokens[i])
                    i += 1
            tokens = out
    return tokens


def merge_compounds(tokens, other_vocab):
    """Join two adjacent words when their concatenation is a word in the OTHER
    transcript ('step' + 'sisters' when the other text has 'stepsisters')."""
    out, i = [], 0
    while i < len(tokens):
        if i + 1 < len(tokens) and tokens[i][0] + tokens[i + 1][0] in other_vocab:
            out.append((tokens[i][0] + tokens[i + 1][0], tokens[i][1] + tokens[i + 1][1]))
            i += 2
        else:
            out.append(tokens[i])
            i += 1
    return out


def adjust(ref, hyp, do_collapse=True, do_merge=True):
    if do_collapse:
        ref, hyp = collapse_repeats(ref), collapse_repeats(hyp)
    if do_merge:
        rv, hv = {t[0] for t in ref}, {t[0] for t in hyp}
        ref, hyp = merge_compounds(ref, hv), merge_compounds(hyp, rv)
    return ref, hyp


# ----------------------------------------------------------------------------
# Edit distance and alignment
# ----------------------------------------------------------------------------


def align(ref, hyp, sub_cost=None):
    """Word-level alignment (Levenshtein). ref/hyp are lists of strings.
    Returns (total_cost, ops) with ops = [(op, ref_idx, hyp_idx)],
    op in C (correct), S (substitution), D (deletion), I (insertion)."""
    if sub_cost is None:
        sub_cost = lambda a, b: 0.0 if a == b else 1.0
    n, m = len(ref), len(hyp)
    d = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        d[i][0] = float(i)
    for j in range(1, m + 1):
        d[0][j] = float(j)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j - 1] + sub_cost(ref[i - 1], hyp[j - 1]),
                          d[i - 1][j] + 1.0, d[i][j - 1] + 1.0)
    ops, i, j, eps = [], n, m, 1e-9
    while i > 0 or j > 0:
        if i > 0 and j > 0 and abs(d[i][j] - (d[i - 1][j - 1] + sub_cost(ref[i - 1], hyp[j - 1]))) < eps:
            ops.append(("C" if ref[i - 1] == hyp[j - 1] else "S", i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i > 0 and abs(d[i][j] - (d[i - 1][j] + 1.0)) < eps:
            ops.append(("D", i - 1, None))
            i -= 1
        else:
            ops.append(("I", None, j - 1))
            j -= 1
    ops.reverse()
    return d[n][m], ops


try:  # optional fast path for character-level distance
    from rapidfuzz.distance import Levenshtein as _RFLev
except ImportError:
    _RFLev = None


def lev_chars(a: str, b: str) -> int:
    if _RFLev is not None:
        return _RFLev.distance(a, b)
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


@lru_cache(maxsize=None)
def soft_sub_cost(a: str, b: str) -> float:
    """Substitution cost in [0,1]. Near-miss spellings/inflections of longer
    words (aunts/aunt, seemed/seems) cost their relative edit distance; anything
    else, including short words (he/she), costs a full error."""
    if a == b:
        return 0.0
    if min(len(a), len(b)) < 4 or abs(len(a) - len(b)) > 2:
        return 1.0
    c = lev_chars(a, b) / max(len(a), len(b))
    return c if c <= 0.34 else 1.0


# ----------------------------------------------------------------------------
# LCS and Jaccard
# ----------------------------------------------------------------------------


def lcs_length(a, b):
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


def jaccard(a: set, b: set):
    return len(a & b) / len(a | b) if (a or b) else float("nan")


# ----------------------------------------------------------------------------
# Parts of speech (optional)
# ----------------------------------------------------------------------------

PENN_CLASS = [("NN", "NOUN"), ("VB", "VERB"), ("MD", "VERB"), ("JJ", "ADJ"),
              ("RB", "ADV"), ("PRP", "PRON"), ("WP", "PRON"), ("DT", "DET"),
              ("PDT", "DET"), ("WDT", "DET"), ("IN", "ADP"), ("CC", "CONJ"),
              ("TO", "PART"), ("RP", "PART"), ("CD", "NUM"), ("UH", "INTJ"),
              ("EX", "PRON"), ("WRB", "ADV")]
CONTENT = {"NOUN", "VERB", "ADJ", "ADV"}
_tagger = None          # None = not loaded yet, False = unavailable


def get_tagger():
    """Return a function tokens -> [pos classes], or None if NLTK is unavailable."""
    global _tagger
    if _tagger is None:
        _tagger = False
        try:
            import nltk
            for attempt in range(2):
                try:
                    nltk.pos_tag(["test"])
                    break
                except LookupError:
                    if attempt:
                        raise
                    for res in ("averaged_perceptron_tagger_eng", "averaged_perceptron_tagger"):
                        nltk.download(res, quiet=True)

            def tag(tokens):
                out = []
                for _, penn in nltk.pos_tag(tokens):
                    cls = next((c for p, c in PENN_CLASS if penn.startswith(p)), "OTHER")
                    out.append(cls)
                return out
            _tagger = tag
        except Exception as e:  # ImportError, LookupError, no internet, ...
            print(f"  note: POS tagging unavailable ({type(e).__name__}); "
                  "install with `pip install nltk`. Continuing without it.")
    return _tagger or None


# ----------------------------------------------------------------------------
# Scoring one pair
# ----------------------------------------------------------------------------


def score_pair(ref_text, hyp_text, a):
    ref_s, hyp_s = tokenize(ref_text), tokenize(hyp_text)
    ref_a, hyp_a = adjust(ref_s, hyp_s, a.collapse_repeats, a.merge_compounds)
    rs, hs = [t[0] for t in ref_s], [t[0] for t in hyp_s]
    ra, ha = [t[0] for t in ref_a], [t[0] for t in hyp_a]

    _, ops_s = align(rs, hs)
    _, ops_a = align(ra, ha)
    soft_cost, _ = align(ra, ha, soft_sub_cost)

    def counts(ops):
        c = Counter(op for op, _, _ in ops)
        return c["S"], c["D"], c["I"], c["C"]

    S, D, I, C = counts(ops_s)
    aS, aD, aI, aC = counts(ops_a)
    n_s, n_a = S + D + C, aS + aD + aC

    ref_str, hyp_str = " ".join(ra), " ".join(ha)
    cd = lev_chars(ref_str, hyp_str)
    lcs = lcs_length(ra, ha)
    p, r = (lcs / len(ha) if ha else 0.0), (lcs / len(ra) if ra else 0.0)
    bigr = lambda w: set(zip(w, w[1:]))

    res = {
        "ref_words": n_s, "hyp_words": len(hs),
        # strict (same column names as before, so existing R scripts still work)
        "correct": C, "substitutions": S, "deletions": D, "insertions": I,
        "wer": (S + D + I) / n_s,
        # adjusted
        "adj_ref_words": n_a, "adj_correct": aC, "adj_substitutions": aS,
        "adj_deletions": aD, "adj_insertions": aI,
        "adj_wer": (aS + aD + aI) / n_a if n_a else float("nan"),
        "soft_wer": soft_cost / n_a if n_a else float("nan"),
        # characters
        "char_edit_distance": cd,
        "cer": cd / len(ref_str) if ref_str else float("nan"),
        "char_similarity": 1 - cd / max(len(ref_str), len(hyp_str), 1),
        # LCS
        "lcs_len": lcs, "lcs_precision": p, "lcs_recall": r,
        "lcs_f1": 2 * p * r / (p + r) if (p + r) else 0.0,
        # Jaccard
        "jaccard_words": jaccard(set(ra), set(ha)),
        "jaccard_bigrams": jaccard(bigr(ra), bigr(ha)),
    }

    # parts of speech, on the adjusted tokens
    ref_pos = hyp_pos = None
    tagger = get_tagger() if a.pos else None
    if tagger:
        ref_pos = tagger([t[1] for t in ref_a])
        hyp_pos = tagger([t[1] for t in hyp_a])
        cont = [(op, ref_pos[i]) for op, i, _ in ops_a if op != "I" and ref_pos[i] in CONTENT]
        func = [(op, ref_pos[i]) for op, i, _ in ops_a if op != "I" and ref_pos[i] not in CONTENT]
        errs = lambda xs: sum(op != "C" for op, _ in xs) / len(xs) if xs else float("nan")
        subs = [(ref_pos[i], hyp_pos[j]) for op, i, j in ops_a if op == "S"]
        res.update({
            "content_words": len(cont), "content_error_rate": errs(cont),
            "function_words": len(func), "function_error_rate": errs(func),
            "sub_same_pos_rate": (sum(r == h for r, h in subs) / len(subs)) if subs else float("nan"),
        })

    return res, (ref_s, hyp_s, ops_s), (ref_a, hyp_a, ops_a, ref_pos, hyp_pos)


def show_alignment(ref, hyp, ops):
    out = []
    for op, i, j in ops:
        if op == "C":
            out.append(ref[i][0])
        elif op == "S":
            out.append(f"[{ref[i][0]}->{hyp[j][0]}]")
        elif op == "D":
            out.append(f"[-{ref[i][0]}]")
        else:
            out.append(f"[+{hyp[j][0]}]")
    return " ".join(out)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def key_of(path: Path):
    return re.sub(r"_(cinderella|whisper)$", "", path.stem)


def pct(x):
    return f"{x:>8.1%}" if x == x else f"{'n/a':>8}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", required=True, help="folder of reference .txt files")
    ap.add_argument("--hyp", required=True, help="folder of Whisper .txt files")
    ap.add_argument("--csv", help="save per-file results to this CSV")
    ap.add_argument("--errors-csv", help="save the word-level alignment (one row per "
                    "word: file, op, ref_word, hyp_word, rel_pos, ref_pos, hyp_pos)")
    ap.add_argument("--errors-mode", choices=["adjusted", "strict"], default="adjusted")
    ap.add_argument("--show-alignment", action="store_true")
    ap.add_argument("--no-collapse-repeats", dest="collapse_repeats", action="store_false")
    ap.add_argument("--no-merge-compounds", dest="merge_compounds", action="store_false")
    ap.add_argument("--no-pos", dest="pos", action="store_false")
    a = ap.parse_args()

    refs = {key_of(p): p for p in Path(a.ref).glob("*.txt")}
    hyps = {key_of(p): p for p in Path(a.hyp).glob("*.txt")}
    shared = sorted(set(refs) & set(hyps))
    for k in sorted(set(refs) ^ set(hyps)):
        print(f"  note: '{k}' only has a {'reference' if k in refs else 'whisper'} file, skipped")
    if not shared:
        raise SystemExit("No matching file pairs found.")

    rows, err_rows, sub_pairs = [], [], Counter()
    pos_tab = {}                                  # class -> Counter of ops (ref side)
    ins_by_pos = Counter()
    for k in shared:
        ref_text = refs[k].read_text(encoding="utf-8")
        hyp_text = hyps[k].read_text(encoding="utf-8")
        if not tokenize(ref_text):
            print(f"  note: '{k}' has an empty reference, skipped")
            continue
        res, strict, adj = score_pair(ref_text, hyp_text, a)
        res["file"] = k
        rows.append(res)
        ref_a, hyp_a, ops_a, ref_pos, hyp_pos = adj

        if a.show_alignment:
            print(f"\n=== {k} (adjusted alignment) ===\n{show_alignment(ref_a, hyp_a, ops_a)}")
            only_r = sorted({t[0] for t in ref_a} - {t[0] for t in hyp_a})
            only_h = sorted({t[0] for t in hyp_a} - {t[0] for t in ref_a})
            print(f"  only in reference: {', '.join(only_r)}\n  only in whisper:   {', '.join(only_h)}")

        # word-level rows + pooled tables
        if a.errors_mode == "strict":
            ref_t, hyp_t, ops = strict
        else:
            ref_t, hyp_t, ops = ref_a, hyp_a, ops_a
        n_ref = sum(op != "I" for op, _, _ in ops) or 1
        consumed = 0
        for op, i, j in ops:
            if op != "I":
                consumed += 1
            rp = ref_pos[i] if (ref_pos and a.errors_mode == "adjusted" and i is not None) else ""
            hp = hyp_pos[j] if (hyp_pos and a.errors_mode == "adjusted" and j is not None) else ""
            err_rows.append([k, op, ref_t[i][0] if i is not None else "",
                             hyp_t[j][0] if j is not None else "",
                             round(consumed / n_ref, 4), rp, hp])
        for op, i, j in ops_a:
            if op == "S":
                sub_pairs[(ref_a[i][0], hyp_a[j][0])] += 1
            if ref_pos:
                if op != "I":
                    pos_tab.setdefault(ref_pos[i], Counter())[op] += 1
                else:
                    ins_by_pos[hyp_pos[j]] += 1

    if not rows:
        raise SystemExit("No comparable file pairs (all references were empty).")

    # ---- per-file table ----
    print(f"\n{'file':<22}{'ref':>6}{'strict':>9}{'adjusted':>9}{'soft':>9}"
          f"{'LCS F1':>9}{'Jacc.w':>9}{'Jacc.bi':>9}{'CER':>9}")
    for r in rows:
        print(f"{r['file']:<22}{r['ref_words']:>6}{pct(r['wer'])}{pct(r['adj_wer'])}"
              f"{pct(r['soft_wer'])}{pct(r['lcs_f1'])}{pct(r['jaccard_words'])}"
              f"{pct(r['jaccard_bigrams'])}{pct(r['cer'])}")

    # ---- pooled summary ----
    tot = lambda key: sum(r[key] for r in rows)
    mean = lambda key: sum(r[key] for r in rows) / len(rows)
    strict_p = (tot("substitutions") + tot("deletions") + tot("insertions")) / tot("ref_words")
    adj_p = (tot("adj_substitutions") + tot("adj_deletions") + tot("adj_insertions")) / tot("adj_ref_words")
    print(f"\nFiles compared: {len(rows)}")
    print(f"Strict WER (pooled):      {strict_p:.1%}   (mean per file {mean('wer'):.1%})")
    print(f"Adjusted WER (pooled):    {adj_p:.1%}   (mean per file {mean('adj_wer'):.1%})")
    print(f"  -> {strict_p - adj_p:+.1%} points of strict WER came from repetitions/compounds, not real errors")
    print(f"Soft WER (mean):          {mean('soft_wer'):.1%}")
    print(f"Character error rate:     {mean('cer'):.1%} (mean)   char similarity {mean('char_similarity'):.1%}")
    print(f"LCS (mean):               precision {mean('lcs_precision'):.1%}, recall {mean('lcs_recall'):.1%}, F1 {mean('lcs_f1'):.1%}")
    print(f"Jaccard (mean):           words {mean('jaccard_words'):.1%}, bigrams {mean('jaccard_bigrams'):.1%}")

    if "content_error_rate" in rows[0]:
        print(f"Error rate, content words: {mean('content_error_rate'):.1%} (mean) | "
              f"function words: {mean('function_error_rate'):.1%} | "
              f"substitutions keeping the same POS: {mean('sub_same_pos_rate'):.1%}")
        print("\nBy part of speech (adjusted alignment, pooled):")
        print(f"  {'POS':<7}{'words':>7}{'correct':>9}{'subst.':>8}{'deleted':>9}{'inserted':>10}")
        for cls, c in sorted(pos_tab.items(), key=lambda kv: -sum(kv[1].values())):
            n = sum(c.values())
            print(f"  {cls:<7}{n:>7}{c['C'] / n:>9.1%}{c['S'] / n:>8.1%}{c['D'] / n:>9.1%}{ins_by_pos[cls]:>10}")

    if sub_pairs:
        print("\nMost common substitutions (reference -> whisper, adjusted):")
        for (r_, h_), n in sub_pairs.most_common(10):
            print(f"  {r_} -> {h_}  x{n}")

    if a.errors_csv:
        with open(a.errors_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["file", "op", "ref_word", "hyp_word", "rel_pos", "ref_pos", "hyp_pos"])
            w.writerows(err_rows)
        print(f"\nSaved {a.errors_csv} ({a.errors_mode} alignment)")

    if a.csv:
        cols = list(dict.fromkeys(["file"] + [c for r in rows for c in r]))
        with open(a.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        print(f"Saved {a.csv}")


if __name__ == "__main__":
    main()
