import csv
import os
import json

INPUT_PATH = "/media02/nthuy/EnTube_preprocessing/EnTube.csv"
DATA_PATH = "/media02/nthuy/EnTube/2021"
OUTPUT_PATH = "/media02/nthuy/EnTube/entube_2021.json"

def main():
    available_videos = []
    # list all filenames in DATA_PATH
    for filename in os.listdir(DATA_PATH):
        if filename.endswith('.mp4'):
            available_videos.append(filename.split('.')[0])
    print(f"Available videos: {len(available_videos)}")
    json_data = []
    cnt = 0
    with open(INPUT_PATH, mode='r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            video_id = row['video_id']
            if video_id is None or video_id == "":
                assert False, f"video_id is None or empty in row: {row}"
            video_id = video_id.strip()
            if video_id in available_videos:
                cnt += 1
                if row['engagement_rate_label'] is None or row['engagement_rate_label'] == "":
                    assert False, f"engagement_rate_label is None or empty in row: {row}"
                json_item = {
                    "video": os.path.join(DATA_PATH, f"{video_id}.mp4"),
                    "label": str(row['engagement_rate_label']),
                    "conversations": [
                        {
                            "from": "human",
                            "value": "<image>\nPredict engagement label.\n<cls>"
                        },
                        {
                            "from": "gpt",
                            "value": f"The engagement label is {row['engagement_rate_label']}."
                        }
                    ]
                }
                json_data.append(json_item)
    print(f"Total extracted videos: {cnt}")
    with open(OUTPUT_PATH, 'w') as f:
            json.dump(json_data, f, indent=4)

if __name__ == "__main__":
    main()
