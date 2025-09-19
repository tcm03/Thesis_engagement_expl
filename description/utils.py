from typing import Dict, List, Any, Tuple, Set
import json
import os
import torch
import torch.distributed as dist

def read_json_data(json_path: str) -> Dict[str, Any]:
    json_dict: Dict[str, Any] = {}
    with open(json_path, 'r') as f:
        json_data = json.load(f)
        assert isinstance(json_data, list), f"Json format must be a list, but got {type(json_data)}"
        for item in json_data:
            filename = os.path.basename(item["video"]).strip()
            if "." in filename:
                filename = filename.split(".")[0]
            del item["conversations"]
            json_dict[filename] = item
    return json_dict

def get_input_prompt() -> str:
    return "<image>\nDescribe this video in detail with predicted engagement level to viewers.\n"

def get_available_json(json_path: str, output_path: str) -> Tuple[Dict[str, Any], Set[str]]:
    avail_json_dict = {} 
    json_name = os.path.basename(json_path)
    if "." in json_name:
        json_name = json_name.split(".")[0]
    output_path = os.path.join(output_path, f"{json_name}_engcaption.json")
    existed_filenames = set()
    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            json_data = json.load(f)
            for item in json_data:
                filename = os.path.basename(item["video"]).strip()
                if "." in filename:
                    filename = filename.split(".")[0]
                existed_filenames.add(filename)
                avail_json_dict[filename] = item

    return avail_json_dict, existed_filenames

def write_checkpoint(
    args,
    json_dict: Dict[str, Any],
    ddp_world_size: int,
    master_process: bool,
):
    # We first read available json data that have been written
    avail_json_dict, existed_filenames = get_available_json(args.json_path, args.output_path)

    # We then aggregate new json data from procs
    short_json_dict = {}
    for k, v in json_dict.items():
        if "conversations" in v:
            short_json_dict[k] = v
    json_dict = short_json_dict
    gathered_json_dict = [None for _ in range(ddp_world_size)]
    dist.all_gather_object(gathered_json_dict, json_dict) # gathered_json_dict [world_size]

    # We finally update existing json data with new json data and write back (on master proc)
    for proc_dict in gathered_json_dict:
        for filename in proc_dict.keys():
            existed_filenames.add(filename)
        avail_json_dict.update(proc_dict)
    if master_process:
        json_name = os.path.basename(args.json_path)
        if "." in json_name:
            json_name = json_name.split(".")[0]
        output_path = os.path.join(args.output_path, f"{json_name}_engcaption.json")
        json_data = []
        for json_item in avail_json_dict.values():
            json_data.append(json_item)
        with open(output_path, 'w') as f:
            json.dump(json_data, f, indent=4)
    return existed_filenames