import os

def clear_generation_folders(folder_paths):
    for folder_path in folder_paths:
        try:
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    if file.endswith(('.json', '.jpg')):
                        file_path = os.path.join(root, file)
                        os.remove(file_path)
                        print(f"Deleted: {file_path}")
        except Exception as e:
            print(f"An error occurred while clearing {folder_path}: {e}")

# List of folder paths to clear
generation_folders = [
    r"D:\ProgrammingF#\Master's thesis\generation0",
    r"D:\ProgrammingF#\Master's thesis\generation1",
    r"D:\ProgrammingF#\Master's thesis\generation2",
    r"D:\ProgrammingF#\Master's thesis\generation3",
    r"D:\ProgrammingF#\Master's thesis\generation4",
    r"D:\ProgrammingF#\Master's thesis\generation5",
    r"D:\ProgrammingF#\Master's thesis\generation6",
    r"D:\ProgrammingF#\Master's thesis\generation7",
    r"D:\ProgrammingF#\Master's thesis\generation8",
    r"D:\ProgrammingF#\Master's thesis\generation9",
    r"D:\ProgrammingF#\Master's thesis\generation10"
]

clear_generation_folders(generation_folders)
