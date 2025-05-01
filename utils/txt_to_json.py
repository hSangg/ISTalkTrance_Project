import json
import re

def convert_txt_to_json(txt_file_path, json_file_path):
    """
    Reads a text file, parses each line, and converts the data into a JSON file.

    Args:
        txt_file_path (str): The path to the input text file.
        json_file_path (str): The path to the output JSON file.
    """
    data = []
    try:
        with open(txt_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                # Use regex to extract start time, end time, speaker, and transcription
                match = re.match(r"(\d+:\d+:\d+)\s(\d+:\d+:\d+)\s(.*?)\s(.*)", line)
                if match:
                    start_time, end_time, speaker_data, transcription = match.groups()
                    # Create a dictionary for each entry
                    entry = {
                        "start_time": start_time,
                        "end_time": end_time,
                        "speaker_data": speaker_data,
                        "transcription": transcription.strip()  # Remove leading/trailing spaces
                    }
                    data.append(entry)
                # Handle lines that don't match the expected format
                elif line.strip():  #check if line is not empty
                    print(f"Skipping line with unexpected format: {line.strip()}")

    except FileNotFoundError:
        print(f"Error: File not found at {txt_file_path}")
        return
    except Exception as e:
        print(f"An error occurred while processing the file: {e}")
        return

    # Write the list of dictionaries to a JSON file
    try:
        with open(json_file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)  # indent for pretty formatting, ensure_ascii to handle non-ascii
        print(f"Successfully converted data to {json_file_path}")
    except Exception as e:
        print(f"An error occurred while writing to the JSON file: {e}")
        return

if __name__ == "__main__":
    # Example usage:
    txt_file = "test_voice/vuiveuncut_conentinkiemtientrenmang/speech.txt"  # Replace with your input .txt file name
    json_file = "test_voice/vuiveuncut_conentinkiemtientrenmang/speech.json" # Replace with your desired output .json file name
    convert_txt_to_json(txt_file, json_file)
