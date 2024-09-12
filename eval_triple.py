from tqdm import tqdm
import os
import openai
from openai import OpenAI
from datetime import datetime
import json
import csv
import random
import argparse

current_date = datetime.now()
date_mmdd = current_date.strftime('%02m%02d')

def eval_results_ai(key, url, filename, error_file) -> list:
    all_count = 0
    correct_count = 0

    client = OpenAI(api_key=key, base_url=url)

    
    with open(filename, "r", encoding="utf-8") as f:
        lines = f.readlines()

        for line in lines:
            tmp = line.strip().split("\t")
            response = tmp[1]
            label = tmp[2]

            all_count += 1
            prompt = ""
            try:
                gpt_res = client.chat.completions.create(
                    messages = [
                        {"role":"system", "content":"The following sentence is the answer to a triple classification question. Summarize the answer. You should answer only \"Yes\" or \"No\" or \"I am not sure\" according to the given message, no other words. "},
                        {"role":"user", "content":response}
                    ],
                    model = "claude-3-haiku-20240307"
                )
                ans = gpt_res.choices[0].message.content
            except Exception as e:
                ans = "not sure"
                print(f"Request failed with error: {e}")
                with open(error_file, "a", encoding="utf-8") as fff:
                    fff.write(line+"\n")

            print("-------\n", ans)
            ans = ans.replace("\n", "").lower()
            # ans = response
            if label == "1" and ans.find("yes")!= -1:
                correct_count += 1
            if label == "-1" and ans.find("not sure") == -1 and (ans.find("not") != -1 or ans.find("no") != -1 or ans.find("n\'t") != -1):
                correct_count += 1

    acc = 1.0 * correct_count /all_count
    return [all_count, correct_count, acc]


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--result_file', type=str, required=True, help='Result of NB-LLM for triple classification')
parser.add_argument('--error_file', type=str, required=True, help='File to save the errors in evaluation')
parser.add_argument('--key', type=str, required=True, help='API key for the evaluation model')
parser.add_argument('--url', type=str, required=True, help='url for the evaluation model')


args = parser.parse_args()

print(eval_results_ai(args.result_file, args.error_file, args.key, args.url))