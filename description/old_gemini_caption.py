import os
import argparse
from google import genai
from google.genai import types, errors
import json
import logging
import time
import concurrent.futures
import threading

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
                    max_output_tokens=512,
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
    "general": "Please describe the video in detail.",
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

int2eng = {
    0: "not engaged",
    1: "neutral",
    2: "engaged"
}

def process_video(args, item, existing_paths, client, model_name, prompt_template):
    video = item["video"]
    if video in existing_paths:
        return None  # skip already-processed

    video_path = os.path.join(args.data_path, video)
    prompt = prompt_template.format(label=int2eng[int(item["label"])])
    try:
        caption = request_caption(client, video_path, prompt, model_name, backoff_factor=30.)
    except Exception as e:
        logger.error(f"[{video}] Failed after retries: {e}")
        return None

    new_item = item.copy()
    new_item['conversations'][0]['value'] = "<image>\nAnalyze the engagement of this video."
    new_item['conversations'][1]['value'] = caption
    logger.info(f"[{video}] Completed.")
    return new_item

def main(args):
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    with open(args.json_path) as f:
        data = json.load(f)

    output_fname = os.path.join(args.output_path, args.output_fname)
    existing = []
    if os.path.exists(output_fname):
        with open(output_fname) as f:
            existing = json.load(f)
    existing_videos = {it['video'] for it in existing}

    results = []
    lock = threading.Lock()

    max_workers = min(32, os.cpu_count() + 4)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as exec:
        futures = {
            exec.submit(process_video, args, item, existing_videos, client, args.model_name, mega_prompt_template): item
            for item in data
        }

        for fut in concurrent.futures.as_completed(futures):
            video = futures[fut]["video"]
            res = fut.result()
            if res:
                with lock:
                    results.append(res)
            if len(results) % args.logging_steps == 0:
                with lock:
                    logger.info(f"Writing {len(results)} new items to disk.")
                    with open(output_fname, 'w') as outf:
                        json.dump(existing + results, outf, indent=4)

    # Final write for any remainders
    final_list = existing + results
    with open(output_fname, 'w') as outf:
        json.dump(final_list, outf, indent=4)
    logger.info(f"Finished. Total new items: {len(results)}")


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