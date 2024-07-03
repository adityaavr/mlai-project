import os
import time
import requests
from threading import Thread, Lock

access_key = "r1HjiV3w0cK3mwo39jtjAOOiVPaY7wu1nloBH4kuUlQ"
global_image_count = 0
request_count = 0
max_requests_per_hour = 50
lock = Lock()


def download_image(image, query, folder_path):
    global global_image_count, request_count

    try:
        img_url = image['urls']['regular']
        download_location = image['links']['download_location']

        # Send a request to the download endpoint
        headers = {
            "Authorization": f"Client-ID {access_key}"
        }
        requests.get(download_location, headers=headers)
        request_count += 1

        img_data = requests.get(img_url).content
        with lock:
            with open(os.path.join(folder_path, f'{query.replace(" ", "_")}_{global_image_count}.jpg'), 'wb') as handler:
                handler.write(img_data)
            global_image_count += 1
        print(f"Downloaded image {global_image_count} for query '{query}'")

    except Exception as e:
        print(f"Could not download image {global_image_count} for query '{query}': {e}")


def download_images(query, folder_path, num_images_per_query, total_images_needed):
    global global_image_count, request_count
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    page = 1
    threads = []
    while global_image_count < total_images_needed:
        if request_count >= max_requests_per_hour:
            print("Reached max requests per hour limit. Sleeping for an hour...")
            time.sleep(3600)  # Sleep for an hour
            request_count = 0

        url = f"https://api.unsplash.com/search/photos?page={page}&query={query}&client_id={access_key}"
        response = requests.get(url)
        request_count += 1

        if response.status_code == 403:  # If rate limit exceeded, sleep for an hour
            print("Rate limit exceeded. Sleeping for an hour...")
            time.sleep(3600)
            request_count = 0
            continue

        data = response.json()
        for image in data['results']:
            if global_image_count >= total_images_needed:
                break
            thread = Thread(target=download_image, args=(image, query, folder_path))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        page += 1
        time.sleep(1)  # Avoid too frequent requests


# Example usage:
object_queries = [
    'household items', 'kitchen utensils', 'tools', 'gadgets',
    'office supplies', 'toys', 'furniture', 'electronics', 'appliances',
    'sports equipment', 'clothing items', 'footwear', 'accessories', 'outdoor gear', 'iphone in hand',
    'laptop on desk', 'books on shelf', 'food items', 'drinks', 'plants', 'animals', 'vehicles',
]

folder_path = 'dataset/unknown'
num_images_per_query = 100
total_images_needed = 1500

for query in object_queries:
    if global_image_count >= total_images_needed:
        break
    download_images(query, folder_path, num_images_per_query, total_images_needed)

print(f"Downloaded {global_image_count} images successfully.")
