import json
import os
from gemini_caption import request_caption
from google import genai
import logging
import random

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

JSON_PATH = "/media02/nthuy/SnapUGC/SnapUGC_0/snapugc0_test_gemini_rawcap.json"
DATA_PATH = "/media02/nthuy/SnapUGC/SnapUGC_0"
OUTPUT_PATH = "/media02/nthuy/SnapUGC/SnapUGC_0/snapugc0_test_gemini_rawcap_fixed.json"
MAX_RETRIES = 18

def main():
    gemini_models = [
        "gemini-2.5-flash-preview-05-20",
        "gemini-2.5-pro-preview-06-05",
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-1.5-flash",
        # "gemini-1.5-flash-8b",
        "gemini-1.5-pro"
    ]
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    with open(JSON_PATH, 'r') as f:
        data = json.load(f)
    new_data = []
    for item in data:
        retries = 0
        while item["conversations"][1]["value"] is None or item["conversations"][1]["value"] == "":
            if retries == 0:
                print(f"START retrying video {item['video']}")
            retries += 1
            if retries > MAX_RETRIES:
                break
            video_path = os.path.join(DATA_PATH, item['video'])
            caption = request_caption(
                client, 
                video_path, 
                prompt="Please describe the video in detail, start with the description right away and do not include prefixes such as 'Here is the description of the video: '.", 
                model_name=random.choice(gemini_models),
                backoff_factor=30
            )
            print(f"For video {item['video']}, got caption: {caption}")
            item["conversations"][1]["value"] = caption
        new_data.append(item)
        if 0 < retries and retries <= 5:
            print(f"Re-processed {item['video']} with {retries} retries")
        elif retries == 6:
            print(f"Failed to process video {item['video']} after {retries} retries")
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(new_data, f, indent=4)
    return

if __name__ == "__main__":
    main()