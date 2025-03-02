from pyannote.audio import Pipeline
from datetime import timedelta
import torchaudio
import os 

hf_token = os.getenv("HF_TOKEN") 

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=hf_token)

# # send pipeline to GPU (when available)
# import torch
# pipeline.to(torch.device("cuda"),)

# apply pretrained pipeline
diarization = pipeline("wav_videos/raw.WAV")

with open("diarization_output.txt", "w") as file:
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        start_time = str(timedelta(seconds=int(turn.start)))
        end_time = str(timedelta(seconds=int(turn.end)))
        result_line = f"{start_time} {end_time} {speaker}\n"
        
        # Print to console
        print(result_line.strip())
        
        # Write to file
        file.write(result_line)