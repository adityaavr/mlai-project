import os
import random


def remove_images(folder_path, percentage=85):
    files = os.listdir(folder_path)

    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']
    images = [file for file in files if os.path.splitext(file)[1].lower() in image_extensions]

    num_images_to_remove = int(len(images) * (percentage / 100))

    images_to_remove = random.sample(images, num_images_to_remove)

    for image in images_to_remove:
        image_path = os.path.join(folder_path, image)
        os.remove(image_path)
        print(f"Removed: {image_path}")


folder_path = '/Users/aditya/PycharmProjects/mlai_project/dataset/train/rottenapples'
remove_images(folder_path)
