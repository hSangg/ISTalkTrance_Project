import os

def time_to_seconds(time_str: str) -> float:
    """Convert time string (h:mm:ss) to seconds."""
    try:
        parts = time_str.split(':')
        if len(parts) == 3:  # h:mm:ss
            hours = float(parts[0])
            minutes = float(parts[1])
            seconds = float(parts[2])
            return hours * 3600 + minutes * 60 + seconds
        else:
            raise ValueError("Invalid time format")
    except ValueError as e:
        print(f"Error converting time: {e}")
        return -1  # Return -1 if invalid time format

def process_script_file(file_path):
    """
    Read a script file, remove underscores from words, and overwrite the file with the modified content.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        # Process each line to check timestamp difference and remove too-close timestamps
        updated_lines = []
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 3:
                continue  # Skip lines that do not follow the expected format
            
            start_time, end_time, speaker = parts
            
            # Convert the start and end time to seconds
            start_seconds = time_to_seconds(start_time)
            end_seconds = time_to_seconds(end_time)
            
            if start_seconds == -1 or end_seconds == -1:
                continue  # Skip if there was an error in time conversion

            # If the start time and end time are too close (chênh lệch <= 1 giây), skip the line
            if abs(end_seconds - start_seconds) <= 1:
                continue  # Skip this line as the start and end time are too close

            # Add the current line to the updated list
            updated_lines.append(line)

        # Overwrite the file with the updated content
        with open(file_path, 'w', encoding='utf-8') as file:
            file.writelines(updated_lines)

        print(f"Processed: {file_path}")
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def process_folders(base_folder):
    """
    Traverse all folders within the base folder and process each `script.txt` file.
    """
    for root, dirs, files in os.walk(base_folder):
        for file in files:
            if file == "script.txt":
                file_path = os.path.join(root, file)
                process_script_file(file_path)

def convert_wav_case(base_folder):
    """
    Traverse all folders and subfolders within the base folder.
    Rename files from `.WAV` to `.wav` if needed.
    """
    for root, dirs, files in os.walk(base_folder):
        for file in files:
            if file.endswith('.WAV'):
                old_path = os.path.join(root, file)
                new_path = os.path.join(root, file[:-4] + '.wav')
                try:
                    os.rename(old_path, new_path)
                    print(f"Renamed: {old_path} -> {new_path}")
                except Exception as e:
                    print(f"Error renaming {old_path}: {e}")

if __name__ == "__main__":
    # Define the base folder containing the train_voice structure
    base_folder = "test_voice"

    # Check if the folder exists
    if os.path.exists(base_folder) and os.path.isdir(base_folder):
        process_folders(base_folder)
    else:
        print(f"The folder '{base_folder}' does not exist.")
    #
    # # Check if the folder exists
    # if os.path.exists(base_folder) and os.path.isdir(base_folder):
    #     convert_wav_case(base_folder)
    # else:
    #     print(f"The folder '{base_folder}' does not exist.")
