import os
import shutil
from PIL import Image


def copy_top_images(source_folder, destination_folder, num_images=100):
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    subfolders = [f for f in os.listdir(source_folder) if os.path.isdir(os.path.join(source_folder, f))]

    for subfolder in subfolders:
        subfolder_path = os.path.join(source_folder, subfolder)
        image_files = [f for f in os.listdir(subfolder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        # image_files.sort()  # Sorting for consistent order

        if len(image_files) > 0:
            destination_subfolder = os.path.join(destination_folder, subfolder)
            os.makedirs(destination_subfolder, exist_ok=True)

            for image_file in image_files[:num_images]:
                source_image_path = os.path.join(subfolder_path, image_file)
                destination_image_path = os.path.join(destination_subfolder, image_file)
                shutil.copy(source_image_path, destination_image_path)




# Specify source and destination folders
source_folder = r''
destination_folder = r''

# Call the function to copy top images from subfolders
copy_top_images(source_folder, destination_folder)
