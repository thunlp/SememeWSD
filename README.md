# SememeWSD

Code and data for the COLING 2020 paper "Try to Substitute: An Unsupervised Chinese Word Sense Disambiguation Method Based on HowNet".
[[Paper]](https://www.aclweb.org/anthology/2020.coling-main.155/)

## Citation
Please cite our paper if you find it helpful.
```
@inproceedings{hou-etal-2020-try,
    title = "Try to Substitute: An Unsupervised {C}hinese Word Sense Disambiguation Method Based on {H}ow{N}et",
    author = "Hou, Bairu  and Qi, Fanchao  and Zang, Yuan  and Zhang, Xurui  and Liu, Zhiyuan  and Sun, Maosong",
    booktitle = "Proceedings of the 28th International Conference on Computational Linguistics",
    year = "2020",
}
```
This repository is mainly contributed by Bairu Hou, Fanchao Qi and Yuan Zang. To run our WSD model or use the WSD dataset, please refer to the following instructions.

## Key Environment
- torch==1.3.1
- torchvision==0.4.2
- transformers==2.8.0
- OpenHowNet==0.0.1a8
Make sure you have run the following codes to complete the installation of `OpenHowNet` before running any codes in this repo.
```python
import OpenHowNet
OpenHowNet.download()
```

## build the Necessary Files:
- Build the vocabulary table either from a corpus or an external knowledge base like HowNet. We use the [Corpus of People Daily](https://opendata.pku.edu.cn/dataset.xhtml?persistentId=doi:10.18170/DVN/SEYRX5). The corpus is already in the `data` directory. To tokenize the corpus and get the necessray file, please use the following command.
```{shell}
mkdir aux_files
python data_util.py
```
- Other corpus will also be fine.The only thing is you need to modify the `tokenize_corpus` function in the `data_util.py` file to process corpus file with different format.

## Load the Dataset
The current available  HowNet-based Chinese WSD [dataset](https://web.eecs.umich.edu/˜mihalcea/senseval/senseval3/tasks.html#ChineseLS)  is based on an outdated version of HowNet that  cannot be found now. To evaluation and further acamedic use, we  build a new and larger HowNet-based Chinese WSD dataset based on the Chinese Word Sense Annotated Corpus used in SemEval-2007 task 5.

You can load the dataset with either `eval` in Python or `json`.

### Load with Python
```python
dataset = []
with open("data/dataset.txt",'r',encoding = 'utf-8') as f:
	for line in f:
		sample = eval(line.stri(p))
		dataset.append(sample)
```
Each line in the file will be transformed to a `dict` data type in Python. The keys of the dict include: 
`context`: A word list of the sentence include a token `<target>` that masks the targeted polysemous word. 
`part-of-speech`: A list of the part-of-speech for each token in the sentence.
`target_word`: The original polysemous word in the sentence masked by `<target>`
`target_position`: The position of the targeted polysemous word in the word list
`target_word_pos`:  The part-of-speech of targeted polysemous word
`sense`: The correct sense of the targeted polysemous word in the context. Represented by a set of sememes from HowNet

### Load with Json
```python
dataset = []
with open("data/dataset.json",'r',encoding='utf-8') as f:
    for line in f:
        sample = json.loads(line.strip())
        dataset.append(sample)
```
You may need to manually transform some datatype if you load with json, such as the `target_word_pos`, which should be an integer. 

You can load the dataset for the follwoing evaluation or in your own research.

## Run the WSD Model on the Dataset
We recommend use cuda to accelerate the inference. Make sure you have generated the necessary files and put the dataset file in the `data/` directory.
```shell
CUDA_VISIBLE_DEVICES=0 python run_model.py
```
The command will test the model on the whole dataset and generate a log file for further evaluation.
## Evaluation
After you get the log file, you can evaluate it with the following command on various metrics. 
```shell
python parse_log.py --model bert
```



