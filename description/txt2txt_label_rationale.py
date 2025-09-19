import json
import os
import copy

DIR_PATH = "/media02/nthuy/SnapUGC/SnapUGC_0"
INPUT_JSON = "snapugc0_test_engcaption.json"
OUTPUT_JSON = "snapugc0_test_engcaption_label_rationale.json"

int2engagement = {
    0: "not engaged",
    1: "neutral",
    2: "engaged"
}

def main():
    with open(os.path.join(DIR_PATH, INPUT_JSON), "r") as f:
        data = json.load(f)
    print(f"Number of input json items: {len(data)}")
    new_data = []
    for item in data:
        item_engagement = copy.deepcopy(item)
        item_engagement['conversations'][0]['value'] = f"[label] <image>\n"
        item_engagement['conversations'][1]['value'] = int2engagement[int(item['label'])]
        new_data.append(item_engagement)
        item_rationale = copy.deepcopy(item)
        item_rationale['conversations'][0]['value'] = f"[rationale] <image>\n"
        new_data.append(item_rationale)
    print(f"Number of output json items: {len(new_data)}")
    with open(os.path.join(DIR_PATH, OUTPUT_JSON), "w") as f:
        json.dump(new_data, f, indent=4)

if __name__ == "__main__":
    main()