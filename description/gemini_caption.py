import os
import argparse
from google import genai
from google.genai import types, errors
import json
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def request_caption(
    client, 
    video_path: str, 
    prompt: str, 
    model_name: str, 
    max_retries: int = 5, 
    backoff_factor: float = 1.0
):
    assert os.path.exists(video_path), f"Video file {video_path} does not exist"
    video_bytes = open(video_path, 'rb').read()

    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=types.Content(
                    parts=[
                        types.Part(
                            inline_data=types.Blob(data=video_bytes, mime_type='video/mp4')
                        ),
                        types.Part(text=prompt)
                    ]
                ),
                config=types.GenerateContentConfig(
                    temperature=0.5,
                    max_output_tokens=256,
                    top_k=5,
                )
            )
            return response.text
        except errors.ServerError as e:
            if attempt == max_retries:
                logging.error(f"Max retries reached. Failed to process {video_path}.")
                raise
            wait_time = backoff_factor * (2 ** (attempt - 1))
            logging.warning(f"Attempt {attempt} failed with error: {e}. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)

prompt_dict = {
    "content": "Please analyze how the video's opening moments and storytelling elements contribute to this level of engagement. Discuss aspects such as the initial hook, pacing, and narrative structure that may influence viewer attention.",
    "emotion": "Please examine the emotional tones conveyed throughout the video. Consider how these emotions might impact viewer engagement, and whether they align with the video's overall message.",
    "visual": "Please evaluate the visual elements, including color schemes, camera angles, and special effects. Discuss how these stylistic choices may affect the video's appeal and viewer retention.",
    "audio": "Please assess the audio components, such as background music, sound effects, and vocal delivery. Analyze how these auditory elements contribute to the video's engagement level.",
    "trend": "Please determine if the content aligns with current social media trends or cultural moments. Discuss how this alignment, or lack thereof, might influence viewer engagement.",
    "interactivity": "Please explore the interactive elements present, such as calls to action, challenges, or prompts for viewer participation. Evaluate how these features may affect engagement.",
    "authenticity": "Please analyze the authenticity and relatability of the content. Consider factors like the creator's presentation style, language, and subject matter in relation to viewer connection.",
    "information": "Please assess the informational or practical value offered. Discuss how the content provides utility or knowledge to the viewer and its potential impact on engagement.",
    "quality": "Please evaluate the technical aspects of production, including video resolution, editing quality, and overall polish. Analyze how these factors may influence viewer engagement."
}

class GeminiPrompter:

    def __init__(self, system_prompt: str, engagement_label: str, category: str):
        assert category in prompt_dict, f"Category {category} not supported"
        assert engagement_label in ["not engaged", "neutral", "engaged"], f"Engagement label {engagement_label} not supported"
        self.system_prompt = system_prompt
        self.engagement_label = engagement_label
        self.category = category
        self._prompt = None

    def __create_template(self):
        base = f"Regarding the video's engagement, a video can be categorized as either not engaged, neutral, or engaged. This video has been labeled as {self.engagement_label}."
        return self.system_prompt + "\n" + base

    def get_prompt(self):
        if self._prompt is None:
            self._prompt = self.__create_template() + "\n" + prompt_dict[self.category]
        return self._prompt



def main(args):
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    # video_path = "/media02/nthuy/SnapUGC/SnapUGC_0/train/train_7/8787be2d403e980fd2737ac42e6f25b7.mp4"
    # prompt_machine = GeminiPrompter(
    #     system_prompt="Generate a continuous paragraph without any other formats or styles that conforms to the following requirements.",
    #     engagement_label="not engaged",
    #     category="quality"
    # )
    # caption = request_caption(
    #     client, 
    #     video_path, 
    #     prompt=prompt_machine.get_prompt(), 
    #     model_name=args.model_name,
    #     backoff_factor=30
    # )
    # print(f"Prompt: {prompt_machine.get_prompt()}\nGenerated: {caption}")
    # return

    with open(args.json_path, 'r') as f:
        data = json.load(f)
    output_fname = os.path.join(args.output_path, args.output_fname)
    if os.path.exists(output_fname):
        with open(output_fname, 'r') as f:
            new_data = json.load(f)
    else:
        new_data = []
    existed_video_paths = [item['video'] for item in new_data]
    cnt: int = 0
    for i, item in enumerate(data):
        if item['video'] in existed_video_paths:
            logger.info(f"Skipping {item['video']}")
            continue
        cnt += 1
        logger.info(f"Processing {item['video']} ({i}/{len(data)})")
        video_path = os.path.join(args.data_path, item['video'])
        caption = request_caption(
            client, 
            video_path, 
            prompt=args.prompt, 
            model_name=args.model_name,
            backoff_factor=30
        )
        new_item = item.copy()
        new_item['conversations'][0]['value'] = f"<image>\nPlease describe the video in detail."
        new_item['conversations'][1]['value'] = caption
        new_data.append(new_item)

        if cnt % args.logging_steps == 0:
            logger.info(f"Processed {cnt} videos")
            with open(output_fname, 'w') as f:
                json.dump(new_data, f, indent=4)
    
    with open(output_fname, 'w') as f:
        json.dump(new_data, f, indent=4)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Video caption generation from Gemini")
    parser.add_argument(
        '--data_path',
        type=str,
        required=True,
        help="Path to video dataset"
    )
    parser.add_argument(
        '--json_path',
        type=str,
        required=True,
        help="Path to metadata file of the video dataset"
    )
    parser.add_argument(
        '--output_path',
        type=str,
        required=True,
        help="Path to output file"
    )
    parser.add_argument(
        '--output_fname',
        type=str,
        required=True,
        help="Name of the output file"
    )
    parser.add_argument(
        '--model_name',
        type=str,
        required=True,
        help="Name of the Gemini model"
    )
    parser.add_argument(
        '--prompt',
        type=str,
        required=True,
        help="Prompt for the model"
    )
    parser.add_argument(
        '--logging_steps',
        type=int,
        required=True,
        help="Logging steps"
    )
    args = parser.parse_args()
    main(args)