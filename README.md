# NB-LLM: Neighborhood-Boosted Large Language Model for Knowledge Graph Completion

This repo introduces the implementation of NB-LLM in our paper.

## Abstract
Large Language Models (LLMs) have demonstrated remarkable prowess in handling Knowledge Graph Completion (KGC) tasks. 
While existing LLM-based methods consider triples in knowledge graphs (KGs) as text sequences to analyze the semantic information, they often overlook the graph's structural information. In this paper, we propose a novel LLM-based method for KGC, termed **N**eighborhood-**B**oosted **L**arge **L**anguage **M**odel (**NB-LLM**). Our approach transforms KG triples into descriptive texts and leverages these descriptions to enhance the LLM's understanding of both semantic and structural information from KG. Specifically, NB-LLM samples the neighbors and relations of each entity within the knowledge graph to assemble multi-granularity local neighborhood information. This process enriches the entity nodes with structural details, enabling LLMs to comprehend relations more effectively and improve reasoning capabilities through fine-tuning. Additionally, we clarify the details of the tasks and problem statements through customized prompts and instructions, which guide LLMs in the reasoning processes and responses to KGC queries. At last, we employ LLMs to summarize the responses succinctly from NB-LLM to get final answers. Notably, our model is fine-tuned based on a significantly smaller data scale than previous works, benefiting from integrating comprehensive neighborhood information and precise prompts and instructions. Experiments on various knowledge graph benchmarks reveal that NB-LLM achieves new SOTA in two KGC tasks (*i.e.*, triple classification and relation prediction), and even surpasses LLMs fine-tuned with up to 10 times the data volume of our experiments.

<div align="center">
<img src="pics\overall_fig3.png" width="800px">
</div>

## Getting Start

### Installing requirement packages

```shell
pip install -r requirements.txt
```

### Data Release

<!-- 由于GitHub对文件大文件的支持并不友好，我们将模型和数据存储在了这里： 。请从链接下载所有文件，并放在该目录下。完成后，完整文件结构如下：--> 



<!-- 其中，./sys_content/ 和 ./instruction/ 包含了我们使用的提示和指令，请根据你的需求指定自行选择或修改。 我们建议将运行后的结果文件存储在./result/  -->
`./sys_content/` and `./instruction/` include the prompts and instructions  used in NB-LLM. Please select or modify them according to your needs. We recommend storing the result files  in  `./result/` .


### How to run

#### 1. Training
<!-- 如果你只是希望复现论文中的结果，你可以直接跳过这一步。训练过程需要借助Llama-factory进行。你可以从这里下载：。首先，你需要构建训练数据集： -->
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




<!-- 然后，阅读llama-factory中的readme文件，将这些训练数据添加到datainfo.json中。请根据你的设备选择合适的batch_size等参数，这些参数通常不会对结果产生显著的影响。 -->
Then, check the README file in **LLaMA-Factory** and add these training data to `dataset_info.json`. Please select an appropriate `batch_size` and other parameters based on your device. 

#### 2. Inference
<!-- 如果你希望复现论文中的结果，你只需要使用我们在/data中给出的模型和测试文件即可。对于其他情况，请指定模型路径 、测试文件路径和system_content -->
If you want to reproduce the results from the paper, you only need to use the model and test files we provide in `./test/`. For other scenarios, please specify the paths for models, test files, `system_content` files and other parameters.

```shell
python llama_rel.py --model_id "path/to/your/model" --test_file "path/to/your/test_file.csv" --system_content "path/to/your/system_content.txt" --top_k 50 --device_map "auto" --temperature 0.1 --top_p 0.75 --num_beams 4 --max_new_tokens 512
```

```shell
python llama_triple.py --model_id "path/to/your/model" --test_file "path/to/your/test_file.csv" --system_content "path/to/your/system_content.txt" --top_k 50 --device_map "auto" --temperature 0.1 --top_p 0.75 --num_beams 4 --max_new_tokens 512
```

#### 3. Evaluation
<!-- 请填入你的API key和相应的URL来使用官方的接口以进行评估。在评估过程中，有可能会由于网络错误等导致某一条样例测试失败，这些失败样例会保存在error_file中。-->
Please enter your API key and the corresponding URL to use the official interface for result evaluation. During the evaluation process, some samples may fail due to network errors or other issues, and these failed samples will be saved in the `error_file`.

```shell
python eval_triple.py --result_file "path/to/your/prediction/results" --error_file "path/to/seve/errors" --key "your_api_key" --url "https://api.example.com/v1"
```

```shell
python eval_rel.py --result_file "path/to/your/prediction/results" --error_file "path/to/seve/errors" --key "your_api_key" --url "https://api.example.com/v1"
```