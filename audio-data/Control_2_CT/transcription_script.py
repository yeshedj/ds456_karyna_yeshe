#!/usr/bin/env python3
"""
chat_to_whisper.py

Convert AphasiaBank / CHAT (.cha) transcripts into plain, Whisper-style text,
keeping ONLY the participant's (PAR) speech from the Cinderella task.

What "Whisper-style" means here:
  * plain prose: no timestamps, speaker tags, %mor/%gra tiers, or CHAT codes
  * normal capitalization and punctuation
  * filled pauses (um, uh), unintelligible markers, gestures, pauses and
    phonological fragments removed (Whisper rarely transcribes these)
  * errors replaced with the intended word, e.g. "twÉªntu@u [: tongue-twisters]"
    becomes "tongue-twisters"
  * the spoken words are otherwise kept verbatim, including repetitions
    (disable with --drop-retracing)

Usage:
    python chat_to_whisper.py kansas03.cha
    python chat_to_whisper.py transcripts/ -o cleaned/
    python chat_to_whisper.py kansas03.cha --gems Cinderella
    python chat_to_whisper.py kansas03-transcript.pdf          # needs pdfplumber

Requires only the standard library (pdfplumber is optional, PDF input only).
"""

import argparse
import re
import sys
from pathlib import Path

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

# Gem (@G:) sections kept by default. "Cinderella_Intro" is the investigator
# explaining the task, so it is excluded; "Cinderella_Extra" holds the
# participant's closing remarks about the task.
DEFAULT_GEMS = ["Cinderella", "Cinderella_Extra"]

# CHAT clitic-style spellings -> what Whisper would normally write
EXPANSIONS = {
    "hasta": "has to",
    "hadta": "had to",
    "hafta": "have to",
    "needta": "need to",
    "oughta": "ought to",
    "supposta": "supposed to",
    "usta": "used to",
    "sorta": "sort of",
    "kinda": "kind of",
    "lotta": "lot of",
    "coulda": "could have",
    "woulda": "would have",
    "shoulda": "should have",
    "dunno": "don't know",
    "lemme": "let me",
    "gimme": "give me",
}

# --------------------------------------------------------------------------
# Reading / parsing CHAT
# --------------------------------------------------------------------------


def load_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        try:
            import pdfplumber
        except ImportError:
            sys.exit("PDF input needs pdfplumber:  pip install pdfplumber")
        with pdfplumber.open(path) as pdf:
            text = "\n".join(page.extract_text() or "" for page in pdf.pages)
        # drop stray page-number lines ("3", "Page 3", "3 of 12") from the export
        text = re.sub(r"(?im)^\s*(?:page\s+)?\d+(?:\s*(?:of|/)\s*\d+)?\s*$", "", text)
        return text
    return path.read_text(encoding="utf-8", errors="replace")


def base_name(path: Path) -> str:
    """kansas03-transcript.pdf -> kansas03  (so files pair up across scripts)."""
    return re.sub(r"[-_ ]?transcripts?$", "", path.stem, flags=re.IGNORECASE)


def merge_lines(text: str):
    """
    Re-join wrapped lines. Real .cha files indent continuation lines with a
    tab; PDF/text exports wrap with no marker. Either way, any line that does
    not start with '*', '%' or '@' continues the previous line. Also splits
    tiers that got glued together (e.g. '...PUNCT*INV: okay').
    """
    text = re.sub(r"(?<=\S)(?=(?:\*[A-Za-z0-9]{3}:|@G:|@Bg:|@Eg:))", "\n", text)
    merged = []
    for line in text.splitlines():
        if not line.strip():
            continue
        if re.match(r"^[*%@]", line):
            merged.append(line.rstrip())
        elif merged:
            merged[-1] += " " + line.strip()
    return merged


def parse_chat(text: str):
    """Return a list of utterances: {speaker, gem, text, mor}."""
    utterances, gem, cur = [], None, None
    for line in merge_lines(text):
        if line.startswith(("@G:", "@Bg:")):
            gem = line.split(":", 1)[1].strip()
            cur = None
        elif line.startswith("@Eg:"):
            gem, cur = None, None
        elif line.startswith("@End"):
            break
        elif line.startswith("*"):
            speaker, _, body = line[1:].partition(":")
            cur = {"speaker": speaker.strip(), "gem": gem,
                   "text": body.strip(), "mor": ""}
            utterances.append(cur)
        elif line.startswith("%mor:") and cur is not None:
            cur["mor"] = line[5:].strip()
    return utterances


# --------------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------------

TIMESTAMP = re.compile(r"\s*\x15?\d+_\d+\x15?\s*$")


def clean_utterance(text: str, keep_retracing: bool = True) -> str:
    t = text.replace("â€“", "-").replace("â€™", "'")

    # trailing media timestamp, e.g. "... 515_1015"
    t = TIMESTAMP.sub("", t)

    # replace mispronounced / garbled words with the intended word:
    #   garbled [: intended]  ->  intended
    t = re.sub(r"\S+\s+\[::?\s*([^\]]+?)\s*\]", r"\1", t)

    # optionally drop self-corrected material: "<a b> [//] c" / "a [/] a"
    if not keep_retracing:
        t = re.sub(r"<[^<>]*>\s*\[//?\]", " ", t)
        t = re.sub(r"\S+\s*\[//?\]", " ", t)

    # trailing-off / interruption terminators -> ellipsis
    t = re.sub(r"\+(?:\.\.\.|//?\.|/\?|\.\.\?)", "...", t)

    # all remaining [ ... ] annotations (error codes, overlaps, [+ exc], ...)
    t = re.sub(r"\[[^\]]*\]", " ", t)

    # events (&=laughs), fillers (&-um), phonological fragments (&+ph), &~
    t = re.sub(r"&[=+\-~]\S*", " ", t)

    # timed pauses (.) (..) (...)
    t = re.sub(r"\(\.+\)", " ", t)

    # unintelligible / untranscribed markers
    t = re.sub(r"\b(?:xxx|yyy|www)\b", " ", t)

    # scope brackets <like this>  (keep the words)
    t = t.replace("<", " ").replace(">", " ")

    # other linkers/terminators: +, +^ +< +"
    t = re.sub(r"(?<!\w)\+[^\s\w]*", " ", t)

    # letters: "o@l" -> "oh" (as in a spoken date/number), other letters -> A, B..
    t = re.sub(r"\bo@l\b", "oh", t)
    t = re.sub(r"\b(\w)@l\b", lambda m: m.group(1).upper(), t)
    t = re.sub(r"@[\w:]+", "", t)               # remaining special-form markers

    # omitted letters "(be)cause", "(o)kay", "an(d)"  -> because, okay, and
    t = re.sub(r"\((\w+)\)", r"\1", t)

    # lengthened syllables "ki:d", "tell:"
    t = re.sub(r"(?<=\w):", "", t)

    # "hasta" -> "has to" etc.
    for src, dst in EXPANSIONS.items():
        t = re.sub(rf"\b{src}\b", dst, t, flags=re.IGNORECASE)

    # ---- punctuation tidy-up ----
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"\s+([,.?!])", r"\1", t)           # no space before punctuation
    t = re.sub(r"([,.?!])(?:\s*,)+", r"\1", t)     # ", ," -> ","
    t = re.sub(r",\s*([.?!])", r"\1", t)           # ",." -> "."
    t = re.sub(r"^[,.\s]+", "", t)                 # leading junk
    t = re.sub(r"([.?!])(?=[A-Za-z])", r"\1 ", t)  # "...word" -> "... word"

    if not re.search(r"[A-Za-z0-9]", t):
        return ""
    if not re.search(r"[.?!]$", t):
        t += "."
    return t


def proper_nouns_from_mor(utterances) -> set:
    """Use the %mor tier's POS tags to find proper nouns (Cinderella, Seuss...)."""
    names = set()
    for u in utterances:
        for lemma in re.findall(r"propn\|([^\s~|]+)", u["mor"]):
            lemma = re.sub(r"-.*$", "", lemma)
            if lemma.isalpha():
                names.add(lemma.lower())
    return names


def apply_casing(text: str, proper_nouns: set) -> str:
    text = re.sub(r"\bi\b", "I", text)
    for name in proper_nouns:
        text = re.sub(rf"\b{re.escape(name)}\b", name.capitalize(), text,
                      flags=re.IGNORECASE)
    # capitalize sentence starts (but not after an ellipsis)
    text = re.sub(r"(^|(?<!\.)[.?!]\s+)([a-z])",
                  lambda m: m.group(1) + m.group(2).upper(), text)
    return text


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------


def process(text: str, gems, speaker="PAR", keep_retracing=True,
            one_per_line=False) -> str:
    wanted = {g.lower() for g in gems}
    utts = [u for u in parse_chat(text)
            if u["speaker"] == speaker and (u["gem"] or "").lower() in wanted]

    nouns = proper_nouns_from_mor(utts)
    cleaned = []
    for u in utts:
        c = clean_utterance(u["text"], keep_retracing=keep_retracing)
        if c:
            cleaned.append(apply_casing(c, nouns))

    return ("\n" if one_per_line else " ").join(cleaned).strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help=".cha/.txt/.pdf files or folders")
    ap.add_argument("-o", "--outdir", default=None,
                    help="output folder (default: next to each input)")
    ap.add_argument("--gems", nargs="+", default=DEFAULT_GEMS,
                    help=f"@G sections to keep (default: {' '.join(DEFAULT_GEMS)})")
    ap.add_argument("--speaker", default="PAR", help="speaker code (default PAR)")
    ap.add_argument("--drop-retracing", action="store_true",
                    help="remove self-corrected words ([//]) and repeats ([/])")
    ap.add_argument("--one-per-line", action="store_true",
                    help="one utterance per line instead of a single paragraph")
    args = ap.parse_args()

    files = []
    for p in map(Path, args.inputs):
        if p.is_dir():
            files += sorted(x for x in p.iterdir()
                            if x.suffix.lower() in {".cha", ".txt", ".pdf"})
        else:
            files.append(p)

    for f in files:
        out = process(load_text(f), args.gems, args.speaker,
                      keep_retracing=not args.drop_retracing,
                      one_per_line=args.one_per_line)
        outdir = Path(args.outdir) if args.outdir else f.parent
        outdir.mkdir(parents=True, exist_ok=True)
        dest = outdir / f"{base_name(f)}_cinderella.txt"
        dest.write_text(out + "\n", encoding="utf-8")
        status = f"{len(out.split())} words" if out else "NO MATCHING SPEECH FOUND"
        print(f"{f.name} -> {dest}  ({status})")


if __name__ == "__main__":
    main()
