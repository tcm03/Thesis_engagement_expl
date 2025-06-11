import openai
import os

# Retrieve the API key from environment variables
api_key = os.getenv("OPENAI_API_KEY")
if api_key is None:
    raise ValueError("API key not found. Please set the OPENAI_API_KEY environment variable.")

client = openai.OpenAI(api_key=api_key)

# Mapping engagement levels to descriptive labels
label_map = {
    0: "not engaged",
    1: "neutral",
    2: "highly engaged",
}

class PromptRewriteError(Exception):
    """Custom exception for prompt rewriting failures."""
    pass

class FewShotExample:
    def __init__(self, user_prompt, engagement_label, rewritten_prompt):
        self.user_prompt = user_prompt
        self.engagement_label = engagement_label
        self.rewritten_prompt = rewritten_prompt

def rewrite_prompt_with_engagement(original_prompt: str, label: int) -> str:
    engagement_desc = label_map.get(label, "unknown engagement level")
    system_prompt = (
        "You are a helpful assistant tasked with rewriting video caption prompts "
        "to make them more personalized based on viewer engagement levels, either not engaged, neutral, or engaged."
    )

    # Define few-shot examples
    example_1 = FewShotExample(
        user_prompt="The video begins with a close-up of a person's gloved hands holding a pair of sunglasses, with a green tray and a blurred background suggesting an indoor setting. The focus then shifts to a green tray with a pair of sunglasses and a small, round object with a red and blue design, indicating a change in the scene or a different moment in time. The camera zooms in on the sunglasses, revealing a detailed view of the brand logo and the reflective surface, with the green tray and the small object no longer in sight. The focus remains on the sunglasses, highlighting their reflective surface and the brand logo, with the background blurred to",
        engagement_label="engaged",
        rewritten_prompt="The video opens with a close-up of gloved hands carefully holding a pair of sunglasses, set against a green tray and a softly blurred indoor background that hints at a controlled, intentional setting. The scene transitions smoothly to focus solely on the green tray, now featuring the sunglasses alongside a small, round object adorned with a striking red and blue design—suggesting a moment of visual intrigue or narrative shift. As the camera draws in closer, the tray and object fade from view, allowing the sunglasses to take center stage. The lens captures the fine details of the brand logo and the sleek, reflective surface, emphasizing craftsmanship and design. With the background remaining softly out of focus, attention is drawn to the product's elegance and visual appeal. Thus, the video is engaged."
    )
    example_2 = FewShotExample(
        user_prompt="The video begins with a close-up of a person's gloved hand slicing through a large, grilled steak on a wooden cutting board, with the steak's juicy texture and the person's focused expression clearly visible. The scene is set in a kitchen environment, indicated by the stainless steel countertop and the presence of a person in the background. As the video progresses, the slicing action continues with the steak being cut into smaller pieces, and the person's hand and the knife moving to different positions, indicating the ongoing cutting process. The background and the person's attire remain consistent throughout these scenes.",
        engagement_label="neutral",
        rewritten_prompt="The video opens with a close-up of a gloved hand carefully slicing through a large, grilled steak on a wooden cutting board, revealing its juicy texture and the precision of the person preparing it. Set in a kitchen environment, the background includes a stainless steel countertop and a figure standing nearby, suggesting a casual cooking setting. As the video continues, the slicing progresses with the steak being cut into smaller portions, and the hand and knife adjusting positions to complete the task. The background and the person's attire remain unchanged, maintaining a steady visual tone throughout the scene. Thus, the video is neutral."
    )
    example_3 = FewShotExample(
        user_prompt="The video features an individual in a dark, patterned suit with a white shirt, engaging in a series of actions with a pen. Initially, the person is seen holding the pen near their face, with their expression and gaze directed towards the camera, suggesting a moment of contemplation or decision-making. As the video progresses, the individual's actions with the pen evolve; it is brought closer to their face, then slightly away, and eventually held horizontally across their face, obscuring it. Throughout these actions, the person's facial expression changes subtly, indicating varying levels of engagement or reaction to the pen's presence. The background remains consistent throughout",
        engagement_label="not engaged",
        rewritten_prompt="The video shows a person dressed in a dark, patterned suit and white shirt performing a few slow, repetitive actions with a pen. At first, the individual holds the pen near their face while looking toward the camera, though there's little variation in movement or expression. As the video continues, the pen is moved slightly—sometimes closer, sometimes farther—before being held horizontally across the face, momentarily covering it. The changes in facial expression are minimal, and the overall pacing remains steady and uneventful. The background stays the same throughout, with no significant shifts in setting or action. Thus, the video is not engaged."
    )

    # Construct the user prompt with few-shot examples
    user_prompt = (
        f"The original prompt is:\n{example_1.user_prompt}\n"
        f"The viewer engagement level is:\n{example_1.engagement_label}.\n"
        f"The engagement-aligned rewritten prompt is:\n{example_1.rewritten_prompt}\n\n"
        f"The original prompt is:\n{example_2.user_prompt}\n"
        f"The viewer engagement level is:\n{example_2.engagement_label}.\n"
        f"The engagement-aligned rewritten prompt is:\n{example_2.rewritten_prompt}\n\n"
        f"The original prompt is:\n{example_3.user_prompt}\n"
        f"The viewer engagement level is:\n{example_3.engagement_label}.\n"
        f"The engagement-aligned rewritten prompt is:\n{example_3.rewritten_prompt}\n\n"
        f"The original prompt is:\n{original_prompt}\n"
        f"The viewer engagement level is:\n{engagement_desc}.\n"
        "The engagement-aligned rewritten prompt is:\n"
    )

    try:
        # Creating a chat completion request
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=1,
        )
        # Extracting the assistant's reply
        rewritten_prompt = completion.choices[0].message.content.strip()
        return rewritten_prompt
    except openai.RateLimitError as e:
        raise PromptRewriteError(f"Failed to rewrite prompt due to rate limit error: {e}")
    except openai.APIConnectionError as e:
        raise PromptRewriteError(f"Failed to rewrite prompt due to API connection error: {e}")
    except openai.Timeout as e:
        raise PromptRewriteError(f"Failed to rewrite prompt due to timeout error: {e}")
    except Exception as e:
        raise PromptRewriteError(f"An unexpected error occurred: {e}")
