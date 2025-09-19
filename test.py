import os
import json
import copy

INPUT_PATH = "/media02/nthuy/SnapUGC/SnapUGC_0/snapugc0_test_explanation.json"
DATASET_PATH = "/media02/nthuy/SnapUGC/SnapUGC_0"
OUTPUT_PATH_1 = "/media02/nthuy/SnapUGC/SnapUGC_0/snapugc0_test_halfexplanation.json"
OUTPUT_PATH_2 = "/media02/nthuy/SnapUGC/SnapUGC_0/snapugc0_test_halfexplanation_cls.json"

mega_prompt_template = (
    "Regarding the video's engagement, a video can be categorized as either not engaged, neutral, or engaged. "
    "This video has been labeled as {label}. Please output exactly ten sentences in one continuous paragraph. "
    "Sentence 1: describe the video in detail. Sentence 2: analyze how the video’s opening moments and storytelling elements contribute to engagement, including the initial hook, pacing, and narrative structure. "
    "Sentence 3: examine the emotional tones conveyed throughout the video, considering how these emotions impact viewer engagement and whether they align with the overall message. "
    "Sentence 4: evaluate the visual elements—such as color schemes, camera angles, and special effects—and discuss how these stylistic choices may affect appeal and viewer retention. "
    "Sentence 5: assess the audio components, like background music, sound effects, and vocal delivery, analyzing how these auditory elements contribute to engagement. "
    "Sentence 6: determine if the content aligns with current social media trends or cultural moments, and discuss how that alignment or lack thereof might influence engagement. "
    "Sentence 7: explore any interactive elements like calls to action, challenges, or prompts for viewer participation and evaluate their impact on engagement. "
    "Sentence 8: analyze the authenticity and relatability of the content, taking into account the creator’s presentation style, language, and subject matter in relation to viewer connection. "
    "Sentence 9: assess the informational or practical value offered, discussing how the content provides utility or knowledge to the viewer and its potential impact on engagement. "
    "Sentence 10: evaluate technical production quality—video resolution, editing quality, overall polish—and analyze how these factors influence engagement. "
    "Do not output anything else—no headers, lists, examples, or additional commentary—only one paragraph of exactly ten sentences."
)

def main():
    with open(INPUT_PATH, "r") as f:
        data = json.load(f)
    new_data = []
    new_data_2 = []
    for item in data:
        response = item["conversations"][1]["value"]
        sentences = response.split(". ")
        picked_sentences = []
        for i in [0, 1, 2, 3, 9]:
            if i < len(sentences):
                picked_sentences.append(sentences[i])
        item["conversations"][1]["value"] = ". ".join(picked_sentences)
        if not item["conversations"][1]["value"].strip().endswith("."):
            item["conversations"][1]["value"] += "."
        new_item = copy.deepcopy(item)
        new_item["conversations"][0]["value"] += "\n<cls>"
        new_data.append(item)
        new_data_2.append(new_item)

    with open(OUTPUT_PATH_1, "w") as f:
        json.dump(new_data, f, indent=4)
    with open(OUTPUT_PATH_2, "w") as f:
        json.dump(new_data_2, f, indent=4)

if __name__ == "__main__":
    main()


