

# TODO! STEP 1
# Create audio dataset with transcript??
# Cut down audio and transcript to just include the cinderella task (garageband)
# Clean up the transcripts to only include sound utterances, no gesture


import os
import whisper
from whisper.utils import get_writer

model = whisper.load_model("base")

# Transcribe with strict settings for aphasic speech
result = model.transcribe(
    "aphasic_speech_audio.mp3",
    condition_on_previous_text=False,
    word_timestamps=True,
    prompt="Uh, so... he went to the, um, b-b-baked shop... [pause] no, bakery. Yeah."
)

output_directory = "."

# 1. Export as TXT (Best for general LLM input)
txt_writer = get_writer("txt", output_directory)
txt_writer(result, "aphasic_speech_audio.mp3")

# 2. Export as SRT (Best if the LLM needs to see timestamps/pauses)
srt_writer = get_writer("srt", output_directory)
srt_writer(result, "aphasic_speech_audio.mp3")

print("Files saved successfully as .txt and .srt!") 







# TODO! STEP 2
# Use whisper to create transcripts
# Start with control group

# TODO! STEP 3
# Evaluate transcripts against the human made transcripts

# TODO! STEP 4
# Feed transcripts to gpt-??
# Generate some sort of summary for each
# Create a sort of concept rule book that we can compare against

# TODO! STEP 5
# Repeat steps 1-4 for one aphasic speech

# TODO! STEP 6
# Evaluate summary/concept against the rule book
