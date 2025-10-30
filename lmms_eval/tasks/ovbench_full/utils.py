from collections import defaultdict
import os
import datetime
import json
# from lmms_eval.tasks._task_utils.file_utils import generate_submission_file
from pathlib import Path
import yaml
import sys, string
from typing import List, Dict, Optional, Union
import re
import PIL
import numpy as np
from loguru import logger as eval_logger

import io
try:
    from petrel_client.client import Client
    client = Client('~/petreloss.conf')
except Exception as e:
    print(f"Failed to initialize Petrel Client: {e}")
    client = None

DATA_LIST = {
    # "ava": "pnorm2:s3://ava/frames/clip/",
    "ava": "/mnt/petrelfs/share_data/zengxiangyu/Tmp/AVA/frames_1fps/",
    "tao": "pnorm2:s3://tao/frames/val/",
    "coin": "shddnew_zxy:s3://COIN/",
    "hirest": "pnorm2:s3://HiREST/videos/",
}

# hf_home = os.getenv("HF_HOME", "./~/.cache/huggingface")
# base_cache_dir = os.path.expanduser(hf_home)

with open(Path(__file__).parent / "_default_template.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)


cache_name = yaml.safe_load("".join(safe_data))["dataset_kwargs"]["cache_dir"]


def ovbench_doc_to_visual(doc, lmms_eval_specific_kwargs=None):
    # cache_dir = os.path.join(base_cache_dir, cache_name)
    cache_dir = ""
    dataset_folder = DATA_LIST[lmms_eval_specific_kwargs["sub_task"]]
    video_path = os.path.join(cache_dir, dataset_folder, doc["video"])
    if os.path.exists(video_path):
        video_path = video_path
    elif os.path.basename(dataset_folder) in ["clevrer", "star"]:
        alternative_video_path = os.path.join(cache_dir, "data0613", dataset_folder, doc["video"])
        if os.path.exists(alternative_video_path):
            video_path = alternative_video_path
        else:
            eval_logger.error(f"Video path: {video_path} does not exist, please check.")
    elif "s3://" not in video_path:
        eval_logger.error(f"Video path: {video_path} does not exist, please check.")

    if "start" in doc:
        start, end = doc['start'], doc['end']
        media_dict = {'start':start, 'end':end, 'video_read_type': 'decord'}
    else:
        media_dict = {'video_read_type': 'decord'}

    
    return [video_path, media_dict]

# def read_frame(video_path):
#     if os.path.exists(video_path):
#         max_frames = len(os.listdir(video_path))
#     else:
#         max_frames = len([k for k in client.list(video_path)])

#     images_group = []
  
#     for frame_index in range(1, max_frames+1):
#         if "s3://" in video_path:
#             img_bytes = client.get(os.path.join(video_path, f"{frame_index:05d}.jpg"))
#             img = PIL.Image.open(io.BytesIO(img_bytes)).convert("RGB")
#         else:
#             img = PIL.Image.open(os.path.join(video_path, f"{frame_index:05d}.jpg")).convert("RGB")
#         images_group.append(img)

#     return images_group
    

def ovbench_frames_doc_to_visual(doc, lmms_eval_specific_kwargs=None):
    # cache_dir = os.path.join(base_cache_dir, cache_name)
    cache_dir = ""
    dataset_folder = DATA_LIST[lmms_eval_specific_kwargs["sub_task"]]
    video_path = os.path.join(cache_dir, dataset_folder, doc["video"])
    if os.path.exists(video_path):
        video_path = video_path
    elif os.path.basename(dataset_folder) in ["clevrer", "star"]:
        alternative_video_path = os.path.join(cache_dir, "data0613", dataset_folder, doc["video"])
        if os.path.exists(alternative_video_path):
            video_path = alternative_video_path
        else:
            eval_logger.error(f"Video path: {video_path} does not exist, please check.")
    elif "s3://" not in video_path:
        eval_logger.error(f"Video path: {video_path} does not exist, please check.")

    # frame_image_list = read_frame(video_path)
    if "start" in doc:
        start, end = doc['start'], doc['end']
        if "fps" in doc:
            fps = doc['fps']
            media_dict = {'start':start, 'end':end, 'fps':fps, 'video_read_type': 'img'} 
        else:
            media_dict = {'start':start, 'end':end, 'video_read_type': 'img'}   
    else:
        media_dict = {'video_read_type': 'img'}

    return [video_path, media_dict]


def ovbench_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    option_prompt = ""
    option_list = doc["candidates"]
    option_letters = string.ascii_uppercase
    for char_index, option in enumerate(option_list):
        option_letter = option_letters[char_index]
        option_prompt += f"{option_letter}. {option}\n"

    full_text = doc["question"] + "\n" + option_prompt + lmms_eval_specific_kwargs["post_prompt"]
    return full_text


def mcq_acc(answer, pred):
    periodStrip = re.compile("(?!<=\d)(\.)(?!\d)")
    commaStrip = re.compile("(\d)(\,)(\d)")
    punct = [";", r"/", "[", "]", '"', "{", "}", "(", ")", "=", "+", "\\", "_", "-", ">", "<", "@", "`", ",", "?", "!"]

    def processPunctuation(inText):
        outText = inText
        for p in punct:
            if (p + " " in inText or " " + p in inText) or (re.search(commaStrip, inText) != None):
                outText = outText.replace(p, "")
            else:
                outText = outText.replace(p, " ")
        outText = periodStrip.sub("", outText, re.UNICODE)
        return outText

    def process(answer):
        option_regex = re.compile(r"^([A-E])\.\s*(.+)$", re.IGNORECASE)
        match = option_regex.match(answer.strip())

        if match:
            # If matched, return the option letter in uppercase
            return match.group(1).upper()
        else:
            # If no match, process the answer as before
            answer = answer.replace("\n", " ")
            answer = answer.replace("\t", " ")
            answer = answer.strip()
            answer = processPunctuation(answer)
            answer = answer.strip("'")
            answer = answer.strip('"')
            answer = answer.strip(")")
            answer = answer.strip("(")
            answer = answer.strip().lower()

            # Try to find any single letter (A-E) in the processed answer
            letter_match = re.search(r"\b([A-E])\b", answer, re.IGNORECASE)
            if letter_match:
                return letter_match.group(1).upper()

            return answer

    pred = process(pred)
    answer = process(answer)

    if pred == answer:
        score = 1
    else:
        score = 0

    return score


def ovbench_process_results(doc, results):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name (in this case ovbench_perception_score), value: metric value
    """
    pred = results[0]

    # Calculate the ground truth option letter
    option_letters = string.ascii_uppercase
    gt_option_letter = None
    for i, candidate in enumerate(doc["candidates"]):
        if candidate == doc["answer"]:
            gt_option_letter = option_letters[i]
            break

    if gt_option_letter is not None:
        # Calculate the score using mcq_acc function
        score = mcq_acc(gt_option_letter, pred)
    else:
        score = 0

    data_dict = {"pred_answer": pred, "gt_answer": gt_option_letter, "score": score, "answer_type": doc["answer_type"], "sub_answer_type":doc["sub_answer_type"]}

    return {"ovbench_accuracy": data_dict}


def ovbench_aggregate_results(results):
    """
    Args:
        results: a list of values returned by process_results
    Returns:
        A score
    """
    total_answered = 0
    total_correct = 0
    answer_type_stats = {}
    sub_answer_type_stats = {}
    for result in results:
        if result["pred_answer"] != "":
            total_answered += 1
            total_correct += result["score"]
            
            if result["answer_type"] not in answer_type_stats:
                answer_type_stats[result["answer_type"]] = {"answered": 0, "correct": 0}
            answer_type_stats[result["answer_type"]]["answered"] += 1
            answer_type_stats[result["answer_type"]]["correct"] += result["score"]
            
            if result["sub_answer_type"] not in sub_answer_type_stats:
                sub_answer_type_stats[result["sub_answer_type"]] = {"answered": 0, "correct": 0}
            sub_answer_type_stats[result["sub_answer_type"]]["answered"] += 1
            sub_answer_type_stats[result["sub_answer_type"]]["correct"] += result["score"]

    print("\n")
    
    print("<<<  Subset instance: ",total_answered, "    Subset correct: ",total_correct ,"    Subset accuracy: ",100 * total_correct / total_answered if total_answered > 0 else 0,"  >>>")
    
    for answer_type, stats in answer_type_stats.items():
        answer_num = stats["answered"]
        correct_num = stats["correct"]
        accuracy = (correct_num / answer_num) * 100 if answer_num > 0 else 0
        print(f"<Task {answer_type}>   instance:{answer_num}   correct:{correct_num}   accuracy:{accuracy:.2f}%")
    
    for sub_answer_type, stats in sub_answer_type_stats.items():
        answer_num = stats["answered"]
        correct_num = stats["correct"]
        accuracy = (correct_num / answer_num) * 100 if answer_num > 0 else 0
        print(f"<Subtask {sub_answer_type}>   instance:{answer_num}   correct:{correct_num}   accuracy:{accuracy:.2f}%")
        
    print("\n")
    
    return 100 * total_correct / total_answered if total_answered > 0 else 0
