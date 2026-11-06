#!/usr/bin/env python3
import argparse
import re
import sys
from pathlib import Path

from transcription_script import (
    DEFAULT_EXCLUDE_PATTERN,
    DEFAULT_GEM_PATTERN,
    base_name,
    gem_matches,
    load_text,
    parse_chat,
)

SUPPORTED_AUDIO_EXTS = {".mp4", ".mp3", ".wav", ".m4a", ".mov", ".mkv", ".flac", ".aac", ".ogg"}
TIMESTAMP_RE = re.compile(r"\x15?(\d+)_(\d+)\x15?")
PADDING_MS = 250


def extract_participant_timestamps(cha_file: Path, gem_include, gem_exclude, target_speaker="PAR"):
    utterance_times = []
    raw_content = load_text(cha_file)

    for entry in parse_chat(raw_content):
        if entry["speaker"] != target_speaker:
            continue
        if not gem_matches(entry["gem"], gem_include, gem_exclude):
            continue

        matches = TIMESTAMP_RE.findall(entry["text"])
        if matches:
            start_ms, end_ms = matches[-1]
            utterance_times.append((int(start_ms), int(end_ms)))

    return utterance_times


def find_audio_match(file_stem: str, audio_directory: Path):
    candidates = sorted(
        f for f in audio_directory.iterdir() if f.suffix.lower() in SUPPORTED_AUDIO_EXTS
    )

    for candidate in candidates:
        if candidate.stem == file_stem:
            return candidate

    for candidate in candidates:
        if candidate.stem.startswith(file_stem):
            return candidate

    return None


def run():
    parser = argparse.ArgumentParser(
        description="Extract and transcribe Cinderella storytelling task using Whisper."
    )
    parser.add_argument("--cha", required=True, help="Input .cha file or directory containing transcripts")
    parser.add_argument("--media", required=True, help="Directory containing corresponding audio/video files")
    parser.add_argument("-o", "--outdir", default="whisper_out", help="Output directory for text files")
    parser.add_argument("--model", default="base", help="Whisper model size (default: base)")
    parser.add_argument("--gem-pattern", default=DEFAULT_GEM_PATTERN, help="Regex for task section labels")
    parser.add_argument("--exclude-pattern", default=DEFAULT_EXCLUDE_PATTERN, help="Regex for labels to ignore")
    parser.add_argument("--per-utterance", action="store_true", help="Transcribe each utterance clip independently")
    parser.add_argument("--whole-file", action="store_true", help="Force full audio file transcription")

    args = parser.parse_args()

    try:
        import whisper
        from whisper.audio import SAMPLE_RATE
    except ImportError:
        sys.exit("Error: OpenAI Whisper is not installed. Run `pip install openai-whisper` first.")

    input_path = Path(args.cha)
    valid_extensions = {".cha", ".pdf", ".txt"}
    
    if input_path.is_dir():
        cha_files = sorted(f for f in input_path.iterdir() if f.suffix.lower() in valid_extensions)
    else:
        cha_files = [input_path]

    media_dir = Path(args.media)
    output_dir = Path(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading Whisper model '{args.model}'...")
    whisper_model = whisper.load_model(args.model)

    def slice_audio(signal, start, end):
        start_idx = max(0, int((start - PADDING_MS) / 1000 * SAMPLE_RATE))
        end_idx = int((end + PADDING_MS) / 1000 * SAMPLE_RATE)
        return signal[start_idx:end_idx]

    for cha_file in cha_files:
        stem_name = base_name(cha_file)
        media_file = find_audio_match(stem_name, media_dir)

        if not media_file:
            print(f"Skipping {cha_file.name}: no matching media found in {media_dir}")
            continue

        timestamps = extract_participant_timestamps(cha_file, args.gem_pattern, args.exclude_pattern)
        if not timestamps:
            print(f"Skipping {cha_file.name}: no valid timestamps found for target task")
            continue

        print(f"Processing {cha_file.name} -> {media_file.name}")
        audio_data = whisper.load_audio(str(media_file))
        total_duration = len(audio_data) / SAMPLE_RATE

        start_sec = min(t[0] for t in timestamps) / 1000
        end_sec = max(t[1] for t in timestamps) / 1000

        print(f"  Duration: {total_duration:.1f}s | Target window: {start_sec:.1f}s to {end_sec:.1f}s ({len(timestamps)} utterances)")

        #Here is a fix for the problem we had with trimming the audio ourselves...we should just give the full audio from now on, even though it takes up a lot of time
        is_pre_cut = args.whole_file or (start_sec >= total_duration)

        if is_pre_cut:
            reason = "--whole-file flag" if args.whole_file else "timestamps exceed audio duration (likely pre-trimmed)"
            print(f"  Transcribing full file ({reason})")
            
            if args.per_utterance:
                print("Notice: --per-utterance ignored for pre-trimmed media.")
                
            transcribed_text = whisper_model.transcribe(audio_data, language="en", fp16=False)["text"].strip()
        else:
            if end_sec > total_duration:
                print("  Warning: Length of clip wanted is longer than length of audio...")

            if args.per_utterance:
                utterance_texts = []
                for start, end in timestamps:
                    clip = slice_audio(audio_data, start, end)
                    if len(clip) >= SAMPLE_RATE * 0.2:
                        res = whisper_model.transcribe(clip, language="en", fp16=False)["text"].strip()
                        if res:
                            utterance_texts.append(res)
                transcribed_text = " ".join(utterance_texts)
            else:
                window_clip = slice_audio(audio_data, start_sec * 1000, end_sec * 1000)
                if len(window_clip) >= SAMPLE_RATE * 0.2:
                    transcribed_text = whisper_model.transcribe(window_clip, language="en", fp16=False)["text"].strip()
                else:
                    transcribed_text = ""

        if not transcribed_text:
            print(f"  Warning: No speech detected in {cha_file.name}. Skipping output.")
            continue

        out_file = output_dir / f"{stem_name}_whisper.txt"
        out_file.write_text(transcribed_text + "\n", encoding="utf-8")
        
        word_count = len(transcribed_text.split())
        print(f"  Saved to {out_file} ({word_count} words)")
        print(f"  Preview: {transcribed_text[:100]}...")


if __name__ == "__main__":
    run()