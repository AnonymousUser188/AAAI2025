import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
from util_func import *
import torch
import json
import argparse
import threading
import time
from accelerate import init_empty_weights, infer_auto_device_map
import transformers
from transformers import AutoConfig, pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from transformers import StoppingCriteria, StoppingCriteriaList
from loguru import logger
from typing import List, Union
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--model_id', type=str, required=True, help='Model ID for the pipeline')
parser.add_argument('--test_file', type=str, required=True, help='Test file to process')
parser.add_argument('--system_content', type=str, default="sys_content/relation_prediction/0.txt", help='System content for the pipeline')
parser.add_argument('--top_k', type=int, default="sys_content/relation_prediction/0.txt", help='System content for the pipeline')
parser.add_argument('--device_map', type=str, default='auto', help='Device map configuration')
parser.add_argument('--temperature', type=float, default=0.1, help='Sampling temperature')
parser.add_argument('--top_p', type=float, default=0.75, help='Top p value for nucleus sampling')
parser.add_argument('--top_k', type=int, default=40, help='Top k value for nucleus sampling')
parser.add_argument('--num_beams', type=int, default=4, help='Number of beams for beam search')
parser.add_argument('--max_new_tokens', type=int, default=512, help='Maximum number of new tokens to generate')
args = parser.parse_args()

model_id = args.model_id
test_file = args.test_file


with open(args.system_content, 'r', encoding='utf-8') as sys_cont_file:
    system_content = sys_cont_file.read()

home_dir = os.path.expanduser('~')
do_int8 = "store_true"
low_cpu_mem_usage = True
port = 12333

res_file = test_file.split(".")[0].replace("data", "output")+"-"+model_id.split("/")[-1]+"-"+str(time.strftime('%m%d_%H%M',time.localtime())) +".csv"

pipe = pipeline(
    "text-generation",
    model=model_id,
    do_sample=False,
    device_map=args.device_map,
    temperature=args.temperature,
    top_p=args.top_p,
    top_k=args.top_k,
    num_beams=args.num_beams,
    max_new_tokens=args.max_new_tokens
)

terminators = [
    pipe.tokenizer.eos_token_id,
    pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]


lines_to_write = []
labels = []
count = 0
correct_count = 0

    
with open(test_file, "r", encoding="utf-8") as f:
    lines = f.readlines()
    # if 'prompt' in str(lines[0]):
    #     lines = lines[1:]
    for line in tqdm(lines[1:]):
        tmp = line.strip().split("\t")        
        prompt = tmp[0]

        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": str(prompt) }
        ]

        response = pipe(
            messages,
            eos_token_id=terminators,            
        )
        # print(response)
        ans = response[0]["generated_text"][-1]["content"].lower()
        count += 1
        label = tmp[1]
        # if ans.find("yes") != -1 and label == "1":
        #     correct_count += 1
        # elif (ans.find("false") != -1 or ans.find("not") != -1 or ans.find("n't") != -1 or ans.find("no") != -1) and label == "-1":
        #     correct_count += 1
        # print(prompt)
        # print("---------------------------")
        # print("\n",correct_count/count,"\n")
        # print("---------------------------")
        # print(ans)
        # print(label)
        lines_to_write.append(prompt+"\t"+ans.replace("\n",".")+"\t"+ label+"\n")
with open(res_file, "w", encoding="utf-8") as f:
    f.writelines(lines_to_write)

# print("# ", model_id)
# print("# ", system_content)
# print("# ", test_file)
# print("# ", res_file)
# print("# ", len(lines), count, correct_count)
# print("# ", correct_count/count)
