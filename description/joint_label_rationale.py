import json
import os

INPUT_JSON = "/media02/nthuy/SnapUGC/SnapUGC_0/snapugc0_test_engcaption.json"
OUTPUT_JSON = "/media02/nthuy/SnapUGC/SnapUGC_0/snapugc0_test_label_rationale_joint.json"

int2eng = {
    0: "not engaged",
    1: "neutral",
    2: "engaged"
}

def main():
    with open(INPUT_JSON, "r") as f:
        data = json.load(f)
    new_data = []
    for item in data:
        item["conversations"][0]["value"] = "[label] [rationale]\n<image>\n"
        eng_label = int2eng[int(item["label"])]
        item["conversations"][1]["value"] = f'{eng_label}\n{item["conversations"][1]["value"]}'
        new_data.append(item)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(new_data, f, indent=4)

if __name__ == "__main__":
    main()