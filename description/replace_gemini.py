import json
import os

FIXED_JSON_PATH = "/media02/nthuy/SnapUGC/SnapUGC_0/snapugc0_train_gemini_rawcap_fixed.json"
RAW_JSON_PATH = "/media02/nthuy/SnapUGC/SnapUGC_0/snapugc0_train_engcaption.json"

def main():
    with open(RAW_JSON_PATH, "r") as f:
        raw_data = json.load(f)
    raw_desc = {}
    for raw_item in raw_data:
        video_path = raw_item["video"]
        desc = raw_item["conversations"][1]["value"]
        raw_desc[video_path] = desc
    with open(FIXED_JSON_PATH, "r") as f:
        fixed_data = json.load(f)
    new_data = []
    for fixed_item in fixed_data:
        video_path = fixed_item["video"]
        assert video_path in raw_desc, f"Video path {video_path} not found in raw description"
        if fixed_item["conversations"][1]["value"] == "" or fixed_item["conversations"][1]["value"] is None:
            print(f"Replacing caption for {video_path}")
            fixed_item["conversations"][1]["value"] = raw_desc[video_path]
        new_data.append(fixed_item)
    with open(FIXED_JSON_PATH, "w") as f:
        json.dump(new_data, f, indent=4)

if __name__ == "__main__":
    main()