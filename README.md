# NB-LLM: Neighborhood-Boosted Large Language Model for Knowledge Graph Completion

This repo introduces the implementation of NB-LLM in our paper.

## Abstract
Large Language Models (LLMs) have demonstrated remarkable prowess in handling Knowledge Graph Completion (KGC) tasks. 
While existing LLM-based methods consider triples in knowledge graphs (KGs) as text sequences to analyze the semantic information, they often overlook the graph's structural information. In this paper, we propose a novel LLM-based method for KGC, termed **N**eighborhood-**B**oosted **L**arge **L**anguage **M**odel (**NB-LLM**). Our approach transforms KG triples into descriptive texts and leverages these descriptions to enhance the LLM's understanding of both semantic and structural information from KG. Specifically, NB-LLM samples the neighbors and relations of each entity within the knowledge graph to assemble multi-granularity local neighborhood information. This process enriches the entity nodes with structural details, enabling LLMs to comprehend relations more effectively and improve reasoning capabilities through fine-tuning.
Additionally, we clarify the details of the tasks and problem statements through customized prompts and instructions, which guide LLMs in the reasoning processes and responses to KGC queries. At last, we employ LLMs to summarize the responses succinctly from NB-LLM to get final answers. Notably, our model is fine-tuned based on a significantly smaller data scale than previous works, benefiting from integrating comprehensive neighborhood information and precise prompts and instructions. Experiments on various knowledge graph benchmarks reveal that NB-LLM achieves new SOTA in two KGC tasks (*i.e.*, triple classification and relation prediction), and even surpasses LLMs fine-tuned with up to 10 times the data volume of our experiments.

<div align="center">
<img src="pics\overall_fig3.png" width=0.9>
</div>

## Getting Start

### Installing requirement packages

```shell
pip install -r requirements.txt
```

### Data Release

We put the models and data on HuggingFace. Please download all files from the link and place them here. 

```shell
git lfs install
git clone https://huggingface.co/Ethan4090/NB-LLM
```


<!--Once completed, the complete file structure should be as follows: . -->

`./sys_content/`  includes the prompts and instructions  used in NB-LLM. Please select or modify them according to your needs. We recommend storing the result files  in  `./result/` .


### How to run

#### 1. Training

If you merely wish to reproduce the results from the paper, you can skip this step directly. The training process requires the use of Llama-factory. You can download it from here: [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory). 

First, you need to construct the training dataset:
```shell
python data/FB13/instruction_triple_train.py
```
```shell
python data/FB13/instruction_rel_train.py
```
```shell
python data/WN11/instruction_triple_train.py
```
```shell
python data/WN11/instruction_rel_train.py
```
```shell
python data/YAGO3-10/instruction_triple_train.py
```
```shell
python data/YAGO3-10/instruction_rel_train.py
```





Then, check the README file in **LLaMA-Factory** and add these training data to `dataset_info.json`. Please select an appropriate `batch_size` and other parameters based on your device. 

#### 2. Inference

If you want to reproduce the results from the paper, you only need to use the model and test files we provide in `./test/`. For other scenarios, please specify the paths for models, test files, `system_content` files and other parameters.

```shell
python llama_rel.py --model_id "path/to/your/model" --test_file "path/to/your/test_file.csv" --system_content "path/to/your/system_content.txt" --top_k 50 --device_map "auto" --temperature 0.1 --top_p 0.75 --num_beams 4 --max_new_tokens 512
```

```shell
python llama_triple.py --model_id "path/to/your/model" --test_file "path/to/your/test_file.csv" --system_content "path/to/your/system_content.txt" --top_k 50 --device_map "auto" --temperature 0.1 --top_p 0.75 --num_beams 4 --max_new_tokens 512
```

#### 3. Evaluation

Please enter your API key and the corresponding URL to use the official interface for result evaluation. During the evaluation process, some samples may fail due to network errors or other issues, and these failed samples will be saved in the `error_file`.

```shell
python eval_triple.py --result_file "path/to/your/prediction/results" --error_file "path/to/save/errors" --key "your_api_key" --url "https://api.example.com/v1"
```

```shell
python eval_rel.py --result_file "path/to/your/prediction/results" --error_file "path/to/seve/errors" --key "your_api_key" --url "https://api.example.com/v1"
```
