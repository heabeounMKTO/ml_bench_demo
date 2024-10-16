import os 

def get_all_image_from_dir(input_path: str):
    image_files = []
    for file in os.listdir(input_path):
        if file.endswith((".jpeg", ".png", ".jpg")):
            image_files.append(file)
    return image_files
