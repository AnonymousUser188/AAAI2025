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

print("=====================================================")

correct_line = []
wrong_line = []
    
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
        if ans.find("yes") != -1 and label == "1":
            correct_count += 1
            correct_line.append(line)

        elif (ans.find("false") != -1 or ans.find("not") != -1 or ans.find("n't") != -1 or ans.find("no") != -1) and label == "-1":
            correct_count += 1
            correct_line.append(line)

        else:
            wrong_line.append(line)   

        print(prompt)
        print("---------------------------")
        print("\n",correct_count/count,"\n")
        print("---------------------------")
        print(ans)
        print(label)
        print("==============================================")
        lines_to_write.append(prompt+"\t"+ans.replace("\n",".")+"\t"+ label+"\n")





with open(res_file, "w", encoding="utf-8") as f:
    f.writelines(lines_to_write)



correct_file = test_file.split(".")[0].replace("data", "test")+"-"+model_id.split("/")[-1]+"-"+str(time.strftime('%m%d_%H%M',time.localtime())) +"-correct.csv"
wrong_file = test_file.split(".")[0].replace("data", "test")+"-"+model_id.split("/")[-1]+"-"+str(time.strftime('%m%d_%H%M',time.localtime())) +"-wrong.csv"
with open(correct_file, "w", encoding="utf-8") as f:
    f.writelines(correct_line)
with open(wrong_file, "w", encoding="utf-8") as f:
    f.writelines(wrong_line)


print("\n\n--------------\n")
print("# ==============================================================")
print("# 模型：", model_id)
print("# 提示：", system_content)
print("# 测试文件：", test_file)
print("# 结果文件：", res_file)
print("# ", len(lines), count, correct_count)
print("# test quick acc:", correct_count/count)







# ================================================================================================
# ================================             FB13             ==================================
# ================================================================================================
# ==============================================================
# 模型： /home/user/chenqizhi/Data/Lora_Model/Llama/Llama-3-8B-Instruct-FB13-triple_neighbor-sft_0723
# 测试文件： data/FB13/test_instructions_llama_neighbor_100.csv
# 结果文件： output/FB13/test_instructions_llama_neighbor_100-Llama-3-8B-Instruct-FB13-triple_neighbor-sft_0723-0806_1611.csv
#  101 100 65
# test quick acc: 0.65

# ==============================================================
# 模型： /home/user/chenqizhi/Data/Lora_plus_Model/Llama/Llama-3-8B-Instruct-wn11-triple-0718
# 测试文件： data/FB13/test_instructions_llama_neighbor_100.csv
# 结果文件： output/FB13/test_instructions_llama_neighbor_100-Llama-3-8B-Instruct-wn11-triple-0718-0808_2111.csv
#  101 100 82
# test quick acc: 0.82

# ==============================================================
# 模型： /home/user/chenqizhi/Data/Model/Llama/Llama-3-8B-Instruct
# 测试文件： data/FB13/test_instructions_llama_neighbor_100.csv
# 结果文件： output/FB13/test_instructions_llama_neighbor_100-Llama-3-8B-Instruct-0808_2258.csv
#  101 100 82
# test quick acc: 0.82
# ==============================================================
# 模型： /home/user/chenqizhi/Data/Model/Llama/Llama-3-8B-Instruct
# 提示： You're an excellent knowledge graph analyst. You should answer yes or no to each question. When you are sure of the answer, give only "yes" or "no" and do not give any other words. When you are not sure about the answer, you can analyze the entity words in the question one by one and give your answer and explanation
# 测试文件： data/FB13/test_instructions_llama_neighbor_100.csv
# 结果文件： output/FB13/test_instructions_llama_neighbor_100-Llama-3-8B-Instruct-0808_2302.csv
#  101 100 76
# test quick acc: 0.76

# ================================================================================================
# ================================             WN11             ==================================
# ================================================================================================

# ==============================================================
# 模型： /home/user/chenqizhi/Data/Lora_plus_Model/Llama/Llama-3-8B-Instruct-wn11-triple-0718
# 测试文件： data/WN11/test_instructions_llama_full-entity_neighbor_100.csv
# 结果文件： output/WN11/pred_instructions_100_llama3-8B-0723_2049.csv
# 101 100 94
# test quick acc: 0.94

# ==============================================================
# 模型： /home/user/chenqizhi/Data/Lora_plus_Model/Llama/Llama-3-8B-Instruct-wn11-triple-0718
# 测试文件： data/WN11/test_instructions_llama_full-entity_neighbor_100.csv
# 结果文件： output/WN11/test_instructions_llama_full-entity_neighbor_100-Llama-3-8B-Instruct-wn11-triple-0718-0806_1635.csv
#  101 100 94
# test quick acc: 0.94



# ==============================================================
# 模型： /home/user/chenqizhi/Data/Pissa_Model/Llama/Llama-3-8B-Instruct-wn11-triple-0718
# 测试文件： data/YAGO3-10/test_llama_triple_neighbor_full_100.csv
# 结果文件： output/YAGO3-10/test_llama_triple_neighbor_full_100-Llama-3-8B-Instruct-wn11-triple-0718-0808_2138.csv
#  100 99 61
# test quick acc: 0.6161616161616161


# ==============================================================
# 模型： /home/user/chenqizhi/Data/Pissa_Model/Llama/Llama-3-8B-Instruct-FB13-triple_neighbor-sft_0718
# 提示： You're an excellent knowledge graph analyst. You should answer yes or no to each question. When you are sure of the answer, give only "yes" or "no" and do not give any other words. When you are not sure about the answer, you can analyze the entity words in the question one by one and give your answer and explanation
# 测试文件： data/WN11/test_instructions_llama_full-entity_neighbor_100.csv
# 结果文件： output/WN11/test_instructions_llama_full-entity_neighbor_100-Llama-3-8B-Instruct-FB13-triple_neighbor-sft_0718-0808_2309.csv
#  101 100 54
# test quick acc: 0.54