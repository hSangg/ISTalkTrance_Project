import os

from pydub import AudioSegment


def parse_timestamp(timestamp):
    minutes, seconds = map(float, timestamp.split(':'))
    return minutes * 60 + seconds

def split_wav_file(wav_path, script_path, output_dir='split_audio'):
    os.makedirs(output_dir, exist_ok=True)

    with open(script_path, 'r') as f:
        script_lines = f.readlines()

    audio = AudioSegment.from_wav(wav_path)

    for i, line in enumerate(script_lines):
        if not line.strip():
            continue
    
        start, end, speaker = line.strip().split()
        start_ms = int(parse_timestamp(start) * 1000)
        end_ms = int(parse_timestamp(end) * 1000)
        segment = audio[start_ms:end_ms]
        output_filename = os.path.join(output_dir, f'{speaker}_{i+1}.wav')
        segment.export(output_filename, format='wav')
        print(f'Exported {output_filename}: {start} to {end} ({speaker})')
    
    print(f'Splitting complete. Files saved in {output_dir}')

wav_path = 'raw.WAV'
script_path = 'script.txt'
split_wav_file(wav_path, script_path)