#!/usr/bin/env python3
"""
run_whisper.py

Transcribe ONLY the Cinderella portion of each recording with OpenAI Whisper,
using the timestamps in the matching CHAT (.cha) file to find where it is.
Output: <cha_stem>_whisper.txt, ready for evaluate_wer.py.

Must sit in the same folder as chat_to_whisper.py (it reuses its parser).

Setup (once):
    pip install openai-whisper
    brew install ffmpeg            # Mac  (Windows: choco install ffmpeg)

Usage:
    python run_whisper.py --cha transcripts/ --media audio/ -o whisper_out
    python run_whisper.py --cha transcripts/ --media audio/ -o whisper_out --model small
    python run_whisper.py --cha transcripts/ --media audio/ -o whisper_out --per-utterance

Modes:
    default          cut one clip from the first to the last participant
                     Cinderella utterance and transcribe it (realistic: Whisper
                     sees the investigator's short "good"/"okay" too)
    --per-utterance  transcribe each participant utterance's clip separately
                     (cleanest comparison; removes investigator speech)
"""

import argparse
import re
import sys
from pathlib import Path

from transcription_script import DEFAULT_GEMS, load_text, parse_chat

MEDIA_EXTS = {".mp4", ".mp3", ".wav", ".m4a", ".mov", ".mkv", ".flac", ".aac", ".ogg"}
TS = re.compile(r"\x15?(\d+)_(\d+)\x15?\s*$")
PAD_MS = 250  # small padding either side of the clip


def participant_times(cha_path: Path, gems, speaker="PAR"):
    """[(start_ms, end_ms), ...] for the participant's utterances in the gems."""
    wanted = {g.lower() for g in gems}
    times = []
    for u in parse_chat(load_text(cha_path)):
        if u["speaker"] != speaker or (u["gem"] or "").lower() not in wanted:
            continue
        m = TS.search(u["text"])
        if m:
            times.append((int(m.group(1)), int(m.group(2))))
    return times


def find_media(stem: str, media_dir: Path):
    """kansas03.cha -> kansas03a.mp4 (exact stem first, then prefix match)."""
    candidates = sorted(p for p in media_dir.iterdir()
                        if p.suffix.lower() in MEDIA_EXTS)
    for p in candidates:
        if p.stem == stem:
            return p
    for p in candidates:
        if p.stem.startswith(stem):
            return p
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cha", required=True, help=".cha file or folder of .cha files")
    ap.add_argument("--media", required=True, help="folder with audio/video files")
    ap.add_argument("-o", "--outdir", default="whisper_out")
    ap.add_argument("--model", default="base",
                    help="tiny, base, small, medium, large (default: base)")
    ap.add_argument("--gems", nargs="+", default=DEFAULT_GEMS)
    ap.add_argument("--per-utterance", action="store_true")
    args = ap.parse_args()

    try:
        import whisper
        from whisper.audio import SAMPLE_RATE
    except ImportError:
        sys.exit("Install Whisper first:  pip install openai-whisper")

    cha_in = Path(args.cha)
    cha_files = sorted(cha_in.glob("*.cha")) if cha_in.is_dir() else [cha_in]
    media_dir, outdir = Path(args.media), Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Loading Whisper model '{args.model}'...")
    model = whisper.load_model(args.model)

    def clip(audio, start_ms, end_ms):
        s = max(0, int((start_ms - PAD_MS) / 1000 * SAMPLE_RATE))
        e = int((end_ms + PAD_MS) / 1000 * SAMPLE_RATE)
        return audio[s:e]

    def transcribe(audio_clip):
        if len(audio_clip) < SAMPLE_RATE * 0.2:   # under 0.2 s: skip
            return ""
        result = model.transcribe(audio_clip, language="en", fp16=False)
        return result["text"].strip()

    for cha in cha_files:
        media = find_media(cha.stem, media_dir)
        if media is None:
            print(f"{cha.name}: no matching audio/video in {media_dir}, skipped")
            continue
        times = participant_times(cha, args.gems)
        if not times:
            print(f"{cha.name}: no timestamped Cinderella speech found, skipped")
            continue

        print(f"{cha.name}: transcribing {media.name} ...")
        audio = whisper.load_audio(str(media))

        if args.per_utterance:
            text = " ".join(t for t in (transcribe(clip(audio, s, e))
                                        for s, e in times) if t)
        else:
            text = transcribe(clip(audio, min(s for s, _ in times),
                                   max(e for _, e in times)))

        dest = outdir / f"{cha.stem}_whisper.txt"
        dest.write_text(text + "\n", encoding="utf-8")
        print(f"  -> {dest} ({len(text.split())} words)")


if __name__ == "__main__":
    main()
