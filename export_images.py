import os
import requests
import pandas as pd
from PIL import Image

# File path to the meme dataset
meme_data_path = r"C:\Users\Vamshi R A\OneDrive\Desktop\AI-based-Meme-Generator\data\memes_data.tsv"

# Function to download images
def download_images(meme_data_path, save_dir=r"E:\SIC_Project2\data\images"):
    # Ensure save directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Read TSV file into a DataFrame
    try:
        meme_data = pd.read_csv(meme_data_path, sep="\t")  # Read as tab-separated values
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    # Download images
    for index, row in meme_data.iterrows():
        image_url = row.get("ImageURL", "")
        if not isinstance(image_url, str) or not image_url.strip():
            print(f"Skipping row {index}: Invalid URL")
            continue

        if not image_url.startswith("http"):
            image_url = "https:" + image_url  # Ensure the URL is complete

        image_name = f"{row.get('HashId', index)}.jpg"  # Fallback to index if HashId is missing
        image_path = os.path.join(save_dir, image_name)

        try:
            response = requests.get(image_url, stream=True)
            if response.status_code == 200:
                with open(image_path, "wb") as f:
                    f.write(response.content)
                print(f"Downloaded: {image_name}")
            else:
                print(f"Failed to download {image_url}: Status {response.status_code}")
        except Exception as e:
            print(f"Error downloading {image_url}: {e}")

# Run the function
download_images(meme_data_path)