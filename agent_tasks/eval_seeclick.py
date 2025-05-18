'''
Derived from https://github.com/njucckevin/SeeClick/blob/d9e29c3c6895c1b2c3e801439fd078dfcbe857ff/agent_tasks/aitw_test.py,
which is originally adapted from https://github.com/google-research/google-research/tree/master/android_in_the_wild
'''

# evaluation on aitw
# This script refer to the official repo of AITW (https://github.com/google-research/google-research/tree/master/android_in_the_wild)
# to calculate the action matching score

import os
import random
import torch
import json
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import AutoPeftModelForCausalLM
from transformers.generation import GenerationConfig
import re
import logging
import ast
import argparse
from PIL import Image
import numpy as np
import datetime
import csv

import action_matching

NULL_ACTION = 0

def get_seeclick_model(model_path, qwen_path, peft_model):
    tokenizer = AutoTokenizer.from_pretrained(qwen_path, trust_remote_code=True)

    if peft_model:
        model = AutoPeftModelForCausalLM.from_pretrained(peft_model, device_map="cuda", trust_remote_code=True).eval()
        model = model.merge_and_unload()
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cuda", trust_remote_code=True, bf16=True).eval()
    model.generation_config = GenerationConfig.from_pretrained(qwen_path, trust_remote_code=True)

    return tokenizer, model

def get_seeclick_response(model, tokenizer, prompt, img_path):
    query = tokenizer.from_list_format([{'image': img_path}, {'text': prompt}, ])
    with torch.no_grad():
        response, history = model.chat(tokenizer, query=query, history=None)
    return response

def print_model_params(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    logging.info(f"Model: {model.__class__.__name__}")
    logging.info(
        f"  trainable params: {trainable_params} || "
        f"all params: {all_param} || "
        f"trainable%: {100 * trainable_params / all_param:.2f}"
    )

# We nullify the action type to 0 if the action is not in the common action list
def action2step_common(step_data):
    action_type = step_data["action_type_id"]

    if action_type == 4:
        if step_data["action_type_text"] == 'click':  # for click action, we calculate midpoint of touch and lift as the click point
            touch_point = step_data["touch"]
            lift_point = step_data["lift"]
            action_type_new = 4
            click_point = [(touch_point[0] + lift_point[0]) / 2, (touch_point[1] + lift_point[1]) / 2]
            click_point = [f"{item:.3f}" for item in click_point]
            click_point = "({},{})".format(click_point[0], click_point[1])
            action = "{{\"action_type\": {}, \"click_point\": {}}}".format(action_type_new, click_point)
        else:
            action_type_new = NULL_ACTION
            action = "{{\"action_type\": {}}}".format(action_type_new)
    elif action_type == 3:
        typed_text = step_data["type_text"]
        action_type_new = 3
        action = "{{\"action_type\": {}, \"typed_text\": \"{}\"}}".format(action_type_new, typed_text)
    else:
        action_type_new = NULL_ACTION
        action = "{{\"action_type\": {}}}".format(action_type_new)

    is_common = (action_type_new != NULL_ACTION)
    return action, is_common

# Compute and log aggregated metrics from evaluation results
def calculate_metrics(results):
    corr_action=0
    corr_type=0
    num_text=0
    corr_text=0
    num_scroll=0
    corr_scroll=0
    num_click=0
    corr_click=0
    num_both_click=0
    corr_both_click=0
    num_wrong_format=0
    num=0
    for episode in results.keys():
        for step in results[episode]:
            corr_action += step['corr_action']
            corr_type += step['corr_type']
            num_text += step['num_text']
            corr_text += step['corr_text']
            num_scroll += step['num_scroll']
            corr_scroll += step['corr_scroll']
            num_click += step['num_click']
            corr_click += step['corr_click']
            num_both_click += step['num_both_click']
            corr_both_click += step['corr_both_click']
            num_wrong_format += step['num_wrong_format']
            num += 1

    if num == 0: num += 1
    if num_text == 0: num_text += 1
    if num_click == 0: num_click += 1
    if num_both_click == 0: num_both_click += 1
    if num_scroll == 0: num_scroll += 1

    logging.info("[Action Acc]: " + str(corr_action/num))
    logging.info("[Type Acc]: " + str(corr_type/num))
    logging.info("[Text Acc]: " + str(corr_text/num_text))
    logging.info("[Num Text]: " + str(num_text))
    logging.info("[Click Acc]: " + str(corr_click/num_click))
    logging.info("[Num Click]: " + str(num_click))
    logging.info("[Num Wrong Format]: " + str(num_wrong_format))
    logging.info("[Num]: " + str(num))

    # for wanb logging
    metrics = {
        "ActionAcc": corr_action / num,
        "TypeAcc": corr_type / num,
        "TextAcc": corr_text / num_text,
        "NumText": num_text,
        "ClickAcc": corr_click / num_click,
        "NumClick": num_click,
        "ScrollAcc": corr_scroll / num_scroll,
        "NumScroll": num_scroll,
        "BothClickAcc": corr_both_click / num_both_click,
        "NumBothClick": num_both_click,
        "NumWrongFormat": num_wrong_format,
        "Num": num,
    }
    return metrics

# Evaluate model prediction against reference action(s)
def evaluate(j, response, action_common):
    output = {k: 0 for k in [
        "corr_action", "corr_type", "num_text", "corr_text", "num_scroll",
        "corr_scroll", "num_click", "corr_click", "num_both_click", "corr_both_click", "num_wrong_format"
    ]}

    try:
        action_pred = action_matching.pred_2_format(ast.literal_eval(response))
    except Exception as error:
        output["num_wrong_format"] = 1
        logging.info("Step: " + str(j) + " wrong format")
        logging.info(f"An exception occurred: {error}")
        logging.info(f"model output: {response}")
        return output

    for action_dict in action_common: # consider all actions in the list
        action_ref = action_matching.action_2_format(action_dict)

        # type accuracy
        corr_type = (action_pred["action_type"] == action_ref["action_type"])
        if corr_type:
            output["corr_type"] = 1

        # click accuracy
        if action_ref["action_type"] == 4:
            output["num_click"] = 1
            if corr_type:
                try: # separate try-except of check_match for each actions in the list
                    annot_position = np.array(
                        [action_dict["annot_position"][i:i + 4] for i in range(0, len(action_dict["annot_position"]), 4)]) # 4 is the number of coords of a boundingbox
                    check_match = action_matching.check_actions_match(action_pred["touch_point"], action_pred["lift_point"],
                                                                    action_pred["action_type"], action_ref["touch_point"],
                                                                    action_ref["lift_point"], action_ref["action_type"],
                                                                    annot_position)
                    if check_match:
                        output["corr_action"] = 1
                        output["corr_click"] = 1

                except Exception as error:
                    output["num_wrong_format"] = 1
                    logging.info("Step: " + str(j) + " wrong format")
                    logging.info(f"An exception occurred: {error}")
                    logging.info(f"model output: {response}")
                    continue

        # text accuracy
        elif action_ref["action_type"] == 3:
            output["num_text"] = 1
            if corr_type:
                if (action_pred["typed_text"] == action_ref["typed_text"]) or (
                        action_pred["typed_text"] in action_ref["typed_text"]) or (
                        action_ref["typed_text"] in action_pred["typed_text"]):
                    output["corr_action"] = 1
                    output["corr_text"] = 1
        else:
            pass

    if output["corr_action"]:
        logging.info("Step: " + str(j) + " right")
    else:
        logging.info("Step: " + str(j) + " wrong")

    return output


def parse_args():
    parser = argparse.ArgumentParser()
    # model
    parser.add_argument('--model_path', type=str, default="cckevinn/SeeClick")
    parser.add_argument('--qwen_path', type=str, default="Qwen/Qwen-VL-Chat")
    parser.add_argument('--peft_model', type=str, default=None)

    # data
    parser.add_argument('--imgs_dir', type=str, required=True)
    parser.add_argument('--test_json_path', type=str, default='../data/aitw_data_test.json')

    # eval config
    parser.add_argument('--multianswer_history_mode', type=str, default='first', help='random, first')
    parser.add_argument('--num_history', type=int, default=4, help='Number of previous actions to use')

    # log
    parser.add_argument('--log_root', type=str, default='./logs/')
    parser.add_argument('--eval_name', type=str, required=True, help='the saved log file name used for evaluation, e.g., aitw_seeclick, aitw_seeclick_monday')
    parser.add_argument('--task', type=str, required=True, help='the test task name, e.g., aitw, amex, monday, windows_mobile')

    return parser.parse_args()

def main():
    # Setup
    TIMESTAMP = datetime.datetime.now().strftime('%Y_%m_%d-%H_%M_%S')
    args = parse_args()
    assert args.num_history > 0

    os.makedirs(args.log_root, exist_ok=True)
    args.log_file_path = os.path.join(args.log_root, f'{args.eval_name}-{args.task}-{TIMESTAMP}.log')
    args.prediction_file_path = os.path.join(args.log_root, f'prediction-{args.eval_name}-{args.task}-{TIMESTAMP}.json')
    args.csv_path = os.path.join(args.log_root, f'evaluation-{args.eval_name}-{args.task}-{TIMESTAMP}.csv')

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(args.log_file_path),
            logging.StreamHandler()
        ]
    )

    os.makedirs(args.log_root, exist_ok=True)

    # Log arguments
    logging.info(f"Start evaluation at {TIMESTAMP}")
    logging.info(f"Log file: {args.log_file_path}")
    logging.info(f"Prediction file: {args.prediction_file_path}")
    logging.info(f"CSV file: {args.csv_path}")

    logging.info(f"Arguments used in {os.path.basename(__file__)}:")
    arg_dict = vars(args)
    max_key_len = max(len(key) for key in arg_dict.keys())
    for key, value in sorted(arg_dict.items()):
        logging.info(f"  {key.ljust(max_key_len)} : {value}")

    # Set random seeds
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(0)

    # Load model and data
    processor, model = get_seeclick_model(args.model_path, args.qwen_path, args.peft_model)
    print_model_params(model)

    imgs_dir = args.imgs_dir
    test_data = json.load(open(args.test_json_path, 'r'))

    # Inference
    prompt_origin = "Please generate the next move according to the ui screenshot, instruction and previous actions. Instruction: {}. Previous actions: {}"
    predictions = {task: {} for task in test_data.keys()}

    # Loop through tasks and episodes
    for task, episodes in test_data.items():

        logging.info("Task: " + task)

        for j, episode in tqdm(enumerate(episodes), total=len(episodes)):
            ep_id = episode[0]['ep_id']
            if ep_id not in predictions[task]:
                predictions[task][ep_id] = []

            history_actions = [] # including the current step
            for step in episode:
                # AitW, AMEX has only one action for each step, but we convert it to a list for consistency
                if 'action_list' not in step: # aitw, amex
                    action_list = [{
                        key: step[key] for key in [
                            "action_type_id", "action_type_text", "annot_position", "touch", "lift", "type_text"
                        ] if key in step
                    }]
                # MONDAY, Windows Mobile has multiple actions for each step that are all considered correct
                else:
                    action_list = step['action_list']

                action_common = [] # for evaluation
                action_common_str = [] # for history
                for action_dict in action_list:
                    action_str, is_common = action2step_common(action_dict)
                    if is_common:
                        action_common.append(action_dict)
                        action_common_str.append(action_str)

                if len(action_common_str):
                    # Append one of the common actions to the history
                    if args.multianswer_history_mode == "random":
                        history_actions.append(str(np.random.choice(action_common_str)))
                    elif args.multianswer_history_mode == "first":
                        history_actions.append(action_common_str[0])
                else:
                    # If all actions are NULL_ACTION, add a NULL_ACTION to the history
                    history_actions.append("{{\"action_type\": {}}}".format(NULL_ACTION))

                # Skip the step if there are no common actions
                if len(action_common) == 0:
                    continue

                img_filename = step["img_filename"] + '.png'
                img_path = os.path.join(imgs_dir, img_filename)
                if not os.path.exists(img_path):
                    raise FileNotFoundError(f"Image not found: {img_path}")

                goal = step["goal"]
                previous_step = ""
                for i, action in enumerate(history_actions[-(args.num_history+1):-1]):
                    previous_step += 'Step' + str(i) + ': ' + action + ". "
                prompt = prompt_origin.format(goal, previous_step)

                # Model inference
                response = None
                try:    # several sample's img dir lead to error, just jump it
                    response = get_seeclick_response(model, processor, prompt, img_path)
                except Exception as error:
                    logging.info("Episode: " + str(ep_id) + ", Step: " + str(j) + f", error: {error}" + f", response: {response}")
                    continue

                prediction = {"filename": img_filename.split("/")[-1], "action_common": action_common, "response": response}
                predictions[task][ep_id].append(prediction)

    with open(args.prediction_file_path, 'w') as fp:
        json.dump(predictions, fp, indent=2)
    print(f"Predictions saved at {args.prediction_file_path}")

    # Evaluate predictions
    results = {task: {} for task in predictions.keys()}
    for task, episodes in predictions.items():
        for ep_id, steps in episodes.items():
            if ep_id not in results[task]:
                results[task][ep_id] = []
            for j, step in enumerate(steps):
                output = evaluate(j, step['response'], step['action_common'])
                results[task][ep_id].append(output)

    eval_dict = {}
    for task in results.keys():
        logging.info("==="*10)
        logging.info(f"Task: {task}")
        logging.info("==="*10)
        eval_dict[task] = calculate_metrics(results[task])

    metric = sum([x["ActionAcc"] for x in eval_dict.values()]) / len(eval_dict)
    logging.info("==="*10)
    logging.info(f"[Avg Score]: {metric}")
    logging.info("==="*10)

    # Save metrics to CSV
    with open(args.csv_path, 'w') as fp:
        rows = [
            {
                "Task": args.task,
                **metrics,
                **vars(args),
            }
            for task, metrics in eval_dict.items()
        ]
        fieldnames = list(dict.fromkeys([
            key
            for row in rows
            for key in row.keys()
        ]))

        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()

        if args.task == "aitw":
            writer.writerows([
                *rows,
                {"Task": "aitw_avg", "ActionAcc": metric, **vars(args)},
            ])
        else:
            writer.writerows([
                *rows,
            ])

    print(f"Metrics saved at {args.csv_path}")

    # Log elapsed time
    END_TIMESTAMP = datetime.datetime.now().strftime('%Y_%m_%d-%H_%M_%S')
    time_elapsed = (
        datetime.datetime.strptime(END_TIMESTAMP, '%Y_%m_%d-%H_%M_%S') -
        datetime.datetime.strptime(TIMESTAMP, '%Y_%m_%d-%H_%M_%S')
    ).total_seconds()
    time_elapsed_formatted = str(datetime.timedelta(seconds=int(time_elapsed)))
    logging.info(f"Time Elapsed: {time_elapsed_formatted} seconds")

if __name__ == "__main__":
    main()