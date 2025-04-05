import os

with open("train_voice_list.txt", "w", encoding="utf-8") as f:
    for folder in os.listdir("train_voice"):
        folder_path = os.path.join("train_voice", folder)

        if os.path.isdir(folder_path):
            f.write(folder_path + "\n")

with open("test_voice_list.txt", "w", encoding="utf-8") as f:
    for folder in os.listdir("test_voice"):
        folder_path = os.path.join("test_voice", folder)

        if os.path.isdir(folder_path):
            f.write(folder_path + "\n")