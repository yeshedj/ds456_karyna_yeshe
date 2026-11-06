#!/usr/bin/env python3
import argparse
import logging
import re
import sys
from pathlib import Path

DEFAULT_GEM_PATTERN = r"cinder"
DEFAULT_EXCLUDE_PATTERN = r"intro"

# Common CHAT shorthand contractions expanded to full forms
EXPANSIONS = {
    "hasta": "has to", "hadta": "had to", "hafta": "have to", "needta": "need to",
    "oughta": "ought to", "supposta": "supposed to", "usta": "used to", "sorta": "sort of",
    "kinda": "kind of", "lotta": "lot of", "coulda": "could have", "woulda": "would have",
    "shoulda": "should have", "dunno": "don't know", "lemme": "let me", "gimme": "give me",
}


def gem_matches(gem, include=DEFAULT_GEM_PATTERN, exclude=DEFAULT_EXCLUDE_PATTERN):
    if not gem:
        return False
    if exclude and re.search(exclude, gem, re.I):
        return False
    return bool(re.search(include, gem, re.I))


def load_text(path: Path) -> str:
    """Read .cha, .txt, or .pdf files into a normalized string."""
    if path.suffix.lower() == ".pdf":
        logging.getLogger("pdfminer").setLevel(logging.ERROR)
        try:
            import pdfplumber
        except ImportError:
            sys.exit("Error: pdfplumber is required for PDF files. Run `pip install pdfplumber`.")
            
        with pdfplumber.open(path) as pdf:
            raw_text = "\n".join(page.extract_text() or "" for page in pdf.pages)
        raw_text = re.sub(r"(?im)^\s*(?:page\s+)?\d+(?:\s*(?:of|/)\s*\d+)?\s*$", "", raw_text)
    else:
        raw_text = path.read_text(encoding="utf-8", errors="replace")

    # Strip PDF web links, print footprints, and dates
    raw_text = re.sub(r"(?i)\S*(?:https?://|talkbank\.org)\S*(?:[ \t]+\d{1,2}/\d{1,2}\b)?", "", raw_text)
    return re.sub(r"(?i)\b\d{1,2}/\d{1,2}/\d{2,4},?[ \t]+\d{1,2}:\d{2}[ \t]*[AP]M", "", raw_text)


def base_name(path: Path) -> str:
    return re.sub(r"[-_ ]?transcripts?$", "", path.stem, flags=re.I)


def parse_chat(text: str):
    """Parse raw CHAT transcript into structured utterance objects."""
    # Split merged lines before headers
    text = re.sub(r"(?<=\S)(?=(?:\*[A-Za-z0-9]{3}:|@G:|@Bg:|@Eg:))", "\n", text)
    
    utterances, current_gem, current_utt = [], None, None
    
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
            
        if line.startswith(("@G:", "@Bg:")):
            current_gem = line.split(":", 1)[1].strip()
            current_utt = None
        elif line.startswith("@Eg:"):
            current_gem = None
            current_utt = None
        elif line.startswith("@End"):
            break
        elif line.startswith("*"):
            speaker, _, body = line[1:].partition(":")
            current_utt = {
                "speaker": speaker.strip(),
                "gem": current_gem,
                "text": body.strip(),
                "mor": ""
            }
            utterances.append(current_utt)
        elif line.startswith("%mor:") and current_utt:
            current_utt["mor"] = line[5:].strip()
        elif not line.startswith("%") and not line.startswith("@") and current_utt:
            current_utt["text"] += " " + line

    return utterances


def clean_utterance(text: str, keep_retracing: bool = True) -> str:
    """Clean CHAT annotations, timestamps, and symbols into standard text."""
    t = text.replace("â€“", "-").replace("â€™", "'")

    # Remove timestamps and web footprints
    t = re.sub(r"\s*[^\w\s.,?!\x27\x22-]{0,2}\d+_\d+[^\w\s.,?!\x27\x22-]{0,2}", " ", t)
    t = re.sub(r"(?i)\S*(?:https?|talkbank)\S*", " ", t)

    # Keep intended word when corrections are present [:: word]
    t = re.sub(r"\S+\s+\[::?\s*([^\]]+?)\s*\]", r"\1", t)

    # Optionally drop self-corrections [/] and [//]
    if not keep_retracing:
        t = re.sub(r"(<[^<>]*>|\S+)\s*\[//?\]", " ", t)

    # Replace CHAT markers
    t = re.sub(r"\+(?:\.\.\.|//?\.|/\?|\.\.\?)", "...", t)  # Cutoffs to ellipses
    t = re.sub(r"\[[^\]]*\]|&[=+\-~]\S*|\(\.+\)|\b(?:xxx|yyy|www)\b", " ", t)  # Annotations/fillers
    t = re.sub(r"[<>]", " ", t)

    # Handle phoneme and letter markers (@l)
    t = re.sub(r"\bo@l\b", "oh", t, flags=re.I)
    t = re.sub(r"\b(\w)@l\b", lambda m: m.group(1).upper(), t)
    t = re.sub(r"@[\w:]+|\(\w+\)|:", "", t)  # Drop remaining special tags, omitted markers, and elongation

    # Expand colloquial spellings
    for src, dst in EXPANSIONS.items():
        t = re.sub(rf"\b{src}\b", dst, t, flags=re.I)

    # Clean up formatting and punctuation spaces
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"\s+([,.?!])", r"\1", t)
    t = re.sub(r"^[,.\s]+", "", t)

    if not re.search(r"[A-Za-z0-9]", t):
        return ""
    if not t.endswith((".", "?", "!")):
        t += "."
    return t


def extract_proper_nouns(utterances) -> set:
    """Extract proper nouns from morphological tier (%mor) for proper casing."""
    names = set()
    for u in utterances:
        for lemma in re.findall(r"propn\|([^\s~|]+)", u["mor"]):
            clean_name = re.sub(r"-.*$", "", lemma)
            if clean_name.isalpha():
                names.add(clean_name.lower())
    return names


def apply_casing(text: str, proper_nouns: set) -> str:
    text = re.sub(r"\bi\b", "I", text)
    for name in proper_nouns:
        text = re.sub(rf"\b{re.escape(name)}\b", name.capitalize(), text, flags=re.I)
    return re.sub(r"(^|(?<!\.)[.?!]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), text)


def process_transcript(text: str, include=DEFAULT_GEM_PATTERN, exclude=DEFAULT_EXCLUDE_PATTERN,
                       speaker="PAR", keep_retracing=True, one_per_line=False) -> str:
    target_utts = [
        u for u in parse_chat(text)
        if u["speaker"] == speaker and gem_matches(u["gem"], include, exclude)
    ]

    proper_nouns = extract_proper_nouns(target_utts)
    cleaned_lines = []

    for u in target_utts:
        cleaned = clean_utterance(u["text"], keep_retracing=keep_retracing)
        if cleaned:
            cleaned_lines.append(apply_casing(cleaned, proper_nouns))

    delimiter = "\n" if one_per_line else " "
    return delimiter.join(cleaned_lines).strip()


def run():
    parser = argparse.ArgumentParser(description="Clean up CHAT transcripts from AphasiaBank.")
    parser.add_argument("inputs", nargs="+", help="Input files or directories (.cha, .txt, .pdf)")
    parser.add_argument("-o", "--outdir", help="Output directory (defaults to input file directory)")
    parser.add_argument("--gem-pattern", default=DEFAULT_GEM_PATTERN, help="Regex pattern for section matches")
    parser.add_argument("--exclude-pattern", default=DEFAULT_EXCLUDE_PATTERN, help="Regex pattern to exclude")
    parser.add_argument("--speaker", default="PAR", help="Speaker label to filter for (default: PAR)")
    parser.add_argument("--drop-retracing", action="store_true", help="Remove self-corrections [/] and [//]")
    parser.add_argument("--one-per-line", action="store_true", help="Output one utterance per line")
    parser.add_argument("--list-gems", action="store_true", help="List all detected @G section tags and exit")

    args = parser.parse_args()

    input_files = []
    for path in map(Path, args.inputs):
        if path.is_dir():
            input_files.extend(sorted(p for p in path.iterdir() if p.suffix.lower() in {".cha", ".txt", ".pdf"}))
        else:
            input_files.append(path)

    if args.list_gems:
        for f in input_files:
            utterances = parse_chat(load_text(f))
            tag_counts = {}
            for u in utterances:
                if u["speaker"] == args.speaker:
                    tag = u["gem"] or "(no gem)"
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1

            print(f"\n{f.name}")
            for tag, count in tag_counts.items():
                matched = "  <-- matches" if gem_matches(tag, args.gem_pattern, args.exclude_pattern) else ""
                print(f"    {tag}: {count} {args.speaker} utterances{matched}")
        return

    for f in input_files:
        raw_text = load_text(f)
        cleaned_text = process_transcript(
            raw_text,
            include=args.gem_pattern,
            exclude=args.exclude_pattern,
            speaker=args.speaker,
            keep_retracing=not args.drop_retracing,
            one_per_line=args.one_per_line
        )

        if not cleaned_text:
            print(f"Skipping {f.name}: No matching speech found.")
            continue

        target_dir = Path(args.outdir) if args.outdir else f.parent
        target_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = target_dir / f"{base_name(f)}_cinderella.txt"
        output_file.write_text(cleaned_text + "\n", encoding="utf-8")
        
        word_count = len(cleaned_text.split())
        print(f"Saved {f.name} -> {output_file} ({word_count} words)")


if __name__ == "__main__":
    run()