import numpy as np 
import pickle
from sklearn.metrics import f1_score


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='bert')
args = parser.parse_args()

model = args.model
# model = 'dense'
# model = 'cip15'
# model = 'rand'

with open(model + "_log.pkl",'rb') as f:
    logger = pickle.load(f)

result = {}

all_labels = []
all_preds = []
noun_macro = []
verb_macro = []
noun_micro = []
verb_micro = []

for word, data in logger.word_dict.items():
    true_label = np.array([x['real_sense'] for x in data])
    select_label = np.array([x['select_sense'] for x in data])

    all_labels.append(true_label)
    all_preds.append(select_label)


    instance_count = np.zeros(np.max(true_label) + 1)
    for label in true_label:
        instance_count[label] += 1
    micro_f1 = f1_score(true_label,select_label,average = "micro")
    macro_f1 = f1_score(true_label,select_label,average = "macro")

    acc = np.sum(true_label == select_label)/len(true_label)
    
    result[word] = {}
    result[word]['micro'] = micro_f1
    result[word]['macro'] = macro_f1
    result[word]['acc'] = acc
    result[word]['num'] = len(true_label)
    if data[0]['pos'] == 'noun':
        noun_micro += [micro_f1] * len(true_label)
        noun_macro += [macro_f1]
    elif data[0]['pos'] == 'verb':
        verb_micro += [micro_f1] * len(true_label)
        verb_macro += [macro_f1]
    else:
        print("error")

all_micro = []
total_item = 0
# all_micro = np.sum([result[x]['micro'] * len(logger.word_dict[x]) for x in list(logger.word_dict.keys())])/

for x in list(logger.word_dict.keys()):
    all_micro.append(result[x]['micro'] * len(logger.word_dict[x]))
    total_item += len(logger.word_dict[x])
all_micro = np.sum(all_micro)/total_item

all_macro = np.mean([x['macro'] for x in result.values()])

result['all'] = {}
result['noun'] = {}
result['verb'] = {}

result['all']['micro'] = all_micro
result['all']['macro'] = all_macro
result['noun']['micro'] = np.mean(noun_micro)
result['noun']['macro'] = np.mean(noun_macro)
result['verb']['micro'] = np.mean(verb_micro)
result['verb']['macro'] = np.mean(verb_macro)



with open(model + "_result.txt",'w',encoding = 'utf-8') as f:
    for word,res in result.items():
        print("word: ",word, res,file = f)



