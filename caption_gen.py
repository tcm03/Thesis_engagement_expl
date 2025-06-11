import sys
sys.path.append('.')
import os
import argparse
import json
from typing import Dict, List, Any

import torch
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset, DataLoader

from engagement_dataset import EngagementDataset, collate_fn

from longvu.builder import load_pretrained_model
from longvu.constants import (
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from longvu.conversation import conv_templates, SeparatorStyle
from longvu.mm_datautils import (
    KeywordsStoppingCriteria,
    process_images,
    tokenizer_image_token,
)

from oai_call import rewrite_prompt_with_engagement, PromptRewriteError
from utils import *
import time
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(filename)s:%(lineno)d - %(funcName)s - %(levelname)s - %(message)s"
)

ddp = int(os.environ.get("RANK", -1)) != -1

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"



def main(args):

    if ddp:
        assert torch.cuda.is_available(), "Distributed training requires CUDA"
        init_process_group(backend="nccl")
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0 # main process for logging, checkpointing, etc.
    else:
        # non-ddp
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        logging.info(f"Using device: {device}")

    if master_process:
        logging.info(f"Begin loading model")
    device_map = {"": device}
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        "./checkpoints/longvu_qwen", None, "cambrian_qwen", device_map=device_map
    )
    if master_process:
        logging.info(f"Model loaded successfully")
    # dirty trick: freeze whole model except for smallest layer to be able to use DDP
    for name, param in model.named_parameters():
        if "mm_projector.0.bias" in name:
            continue
        param.requires_grad = False
    # model.to(device)
    model.eval()
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module if ddp else model
    if master_process:
        logging.info(f"Model wrapped in DDP")
    

    raw_json_dict = read_json_data(args.json_path)
    json_dict, existed_filenames = get_available_json(args.json_path, args.output_path)

    if master_process:
        logging.info(f"Begin loading dataset")
    dataset = EngagementDataset(
        data_path=args.data_path, 
        json_path=args.json_path, 
        image_processors=image_processor,
        existed_filenames=existed_filenames,
    )
    # Set up DistributedSampler (if in DDP mode) for automatic data splitting
    data_sampler = DistributedSampler(dataset, shuffle=False) if ddp else None
    dataloader = DataLoader(
        dataset,
        batch_size=args.per_device_batch_size, 
        sampler=data_sampler,
        collate_fn=collate_fn,
    )
    if master_process:
        logging.info(f"Dataset loaded successfully")

    qs = "Describe this video in detail."
    qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
    conv = conv_templates["qwen"].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    
    num_steps: int = 0
    for batch_idx, (filenames, videos, image_sizes, labels) in enumerate(dataloader):

        assert isinstance(videos, list) and len(videos) == 2 and isinstance(videos[0], list) and isinstance(videos[1], list) and len(videos[0]) == len(videos[1]), "Must be videos lists for two vision encoders"
        # if master_process:
        logging.info(f"On {device} - Processing batch {batch_idx + 1}/{len(dataloader)} - filenames: {filenames}")
        filename = str(filenames[0]).strip()
        if filename in existed_filenames:
            logging.info(f"Filename {filename} already exists, skipping...")
            continue
        
        batch_size = len(videos[0])
        batch_input_ids = input_ids.repeat(batch_size, 1) # repeat batch_size times along first dim
        
        with torch.inference_mode():
            output_ids = raw_model.generate(
                batch_input_ids,
                images=videos,
                image_sizes=image_sizes,
                do_sample=False,
                temperature=0.2,
                max_new_tokens=128,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )
        preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        preds = [pred.strip() for pred in preds]
        while True:
            try:
                rew_pred = rewrite_prompt_with_engagement(preds[0], int(labels[0]))
                assert filename in raw_json_dict, f"Filename {filename} not found in raw json dict"
                json_dict[filename] = raw_json_dict[filename]
                json_dict[filename]["conversations"] = [
                    {
                        "from": "human",
                        "value": get_input_prompt()
                    },
                    {
                        "from": "gpt",
                        "value": rew_pred
                    }
                ]
                break
            except PromptRewriteError as e:
                logging.info(f"Prompt rewrite failed: {e}")
                logging.info(f"Retrying in 10 seconds...")
                time.sleep(10)
        num_steps += 1
        if num_steps >= args.ckpt_num_steps or batch_idx == len(dataloader) - 1:
            # write back
            dist.barrier()
            if master_process:
                logging.info(f"After {args.ckpt_num_steps} steps, writing checkpoint...")
            existed_filenames = write_checkpoint(
                args,
                json_dict,
                ddp_world_size,
                master_process
            )
            num_steps = 0
        
        torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed Caption Generation")
    parser.add_argument(
        "--model_path", 
        type=str, 
        required=True, 
        help="Path to the model checkpoint"
    )
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
        '--ckpt_num_steps',
        type=int,
        default=100,
        help='Number of steps to write checkpoint'
    )
    parser.add_argument(
        '--per_device_batch_size',
        type=int,
        default=1,
        help='Per-device batch size for running'
    )
    args = parser.parse_args()
    # check if data_paths, json_paths, and output_paths have the same length
    # if len(args.data_paths) != len(args.json_paths) or len(args.data_paths) != len(args.output_paths):
    #     raise ValueError("data_paths, json_paths, and output_paths must have the same length")
    main(args)