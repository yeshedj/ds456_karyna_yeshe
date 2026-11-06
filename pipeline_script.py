# TODO: Step 1
# Put together the audio files and transcripts
# Trim the audio and transcript so they only cover the Cinderella task (GarageBand)
# Clean up the transcripts so they only include spoken words, not gestures


import os
import run_whisper
from whisper.utils import get_writer


model = run_whisper.load_model("base")

# Transcribe the audio using settings for aphasic speech
result = model.transcribe(
    "aphasic_speech_audio.mp3",
    condition_on_previous_text=False,
    word_timestamps=True,
    prompt="Uh, so... he went to the, um, b-b-baked shop... [pause] no, bakery. Yeah."
)

output_directory = "."

# Save a plain text transcript
txt_writer = get_writer("txt", output_directory)
txt_writer(result, "aphasic_speech_audio.mp3")

# Save an SRT file too, so the timestamps are available
srt_writer = get_writer("srt", output_directory)
srt_writer(result, "aphasic_speech_audio.mp3")

print("Transcript files saved as .txt and .srt!")


# TODO: Step 2
# Use Whisper to make transcripts
# Start with the control group


# TODO: Step 3
# Compare the Whisper transcripts with the human-made transcripts


# TODO: Step 4
# Send the transcripts to GPT
# Make a summary for each one
# Put together a set of concepts/rules that the summaries can be compared to


# TODO: Step 5
# Repeat steps 1-4 with one aphasic speech sample


# TODO: Step 6
# Compare the summaries and concepts with the rule set
