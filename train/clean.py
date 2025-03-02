import re
from datetime import datetime

def time_to_seconds(time_str):
    """Convert HH:MM:SS time format to seconds."""
    h, m, s = map(int, time_str.split(':'))
    return h * 3600 + m * 60 + s

def process_diarization_file(input_file, output_file):
    """
    Processes the diarization file by:
    1. Removing rows where start time equals end time.
    2. Removing segments that are 1 second or shorter.
    3. Writing cleaned data to a new file.
    """
    segments = []

    # Read the input file
    with open(input_file, "r") as file:
        for line in file:
            match = re.match(r"(\d{1,2}:\d{2}:\d{2}) (\d{1,2}:\d{2}:\d{2}) (\w+)", line.strip())
            if match:
                start, end, speaker = match.groups()
                if start != end and time_to_seconds(end) - time_to_seconds(start) > 1:
                    segments.append({"start": start, "end": end, "speaker": speaker})
                else:
                    print(f"Skipping invalid/short segment: {line.strip()}")
            else:
                print(f"Invalid line (skipped): {line.strip()}")

    # Sort by start time
    segments.sort(key=lambda x: time_to_seconds(x["start"]))

    # Write cleaned data to the output file
    with open(output_file, "w") as file:
        for seg in segments:
            file.write(f"{seg['start']} {seg['end']} {seg['speaker']}\n")
    
    print(f"Cleaned data has been written to {output_file}")


input_file = "diarization_output.txt"  # Replace with your input file
output_file = "script.txt"  # Replace with your desired output file
process_diarization_file(input_file, output_file)
