import pickle
import numpy as np
import OpenHowNet
from keras.preprocessing.text import Tokenizer
import os

def tokenize_corpus(path = 'data/corpus.txt'):
    """Tokenizes a text file."""
    assert os.path.exists(path)
    # Add words to the dictionary
    sentence_list = []
    pos_list = []
    with open(path, 'r', encoding="utf-8") as f:
        tokens = 0
        for line in f:
            items = line.split()[1:]
            words = [x.split('/')[0] for x in items]
            poses = [x.split('/')[1] for x in items]
            sentence_list.append(words)
            pos_list.append(poses)
    return sentence_list,pos_list

def build_vocab():
    sentence_list,_ = tokenize_corpus()
    vocab_dict = {}
    for sentence in sentence_list:
        words = sentence
        for word in words:
            if word not in vocab_dict:
                vocab_dict[word] = 1  ##[word,pos]
            else:
                vocab_dict[word] += 1
    print("vocabulary table size: %d"%(len(vocab_dict)))
    return vocab_dict

def gen_sem_dict(vocab_dict):
    check_pos_list = ['noun', 'verb', 'adj', 'adv']
    word_pos = {}
    word_sem = {}
    hownet_dict = OpenHowNet.HowNetDict()
    for word,count in vocab_dict.items():
        word_pos[word] = []
        word_sem[word] = {}
        tree = hownet_dict.get_sememes_by_word(word,structured=True,lang='zh',merge= False)
        sememes = hownet_dict.get_sememes_by_word(word,structured= False,lang='zh',merge= False)
        sem_list = [x['sememes'] for x in sememes]
        pos_list = [x['word']['ch_grammar'] for x in tree]
        assert len(pos_list) == len(sem_list),"%d, %d"%(len(sem_list),len(pos_list))
        valid_pos = []
        for i in range(len(pos_list)):
            sem = sem_list[i]
            pos = pos_list[i]
            if pos in check_pos_list:
                valid_pos.append(pos)
                if pos not in word_sem[word]:
                    word_sem[word][pos] = [sem]
                else:
                    if sem not in word_sem[word][pos]:
                        word_sem[word][pos].append(sem)
        word_pos[word] = set(valid_pos)
    return word_sem,word_pos

def gen_id2sem(word_sem):
    id_to_sem = {}
    for word, pos2sems in word_sem.items():
        id_to_sem[word] = {}
        for pos,sem_list in pos2sems.items():
            id_to_sem[word][pos] = {}
            for i in range(len(sem_list)):
                sem_set = sem_list[i]
                id_to_sem[word][pos][i] = sem_set
    return id_to_sem

def compare_list(list1,list2):
    for item in list1:
        assert type(item) == set,'wrong type!'
        if item in list2:
            return True
    return False

def add_word(word,vocab_list,word_candidate,id_to_sem,word_sem,word_pos):
    word_candidate[word] = {}
    if len(word_pos[word]) == 0:
        return

    orig_word_pos = word_pos[word]
    for pos in orig_word_pos:
        word_candidate[word][pos] = {}
    for sub_word in vocab_list:
        sub_word_pos = word_pos[sub_word]
        if len(sub_word_pos&orig_word_pos) == 0:
            continue
        common_pos = sub_word_pos&orig_word_pos
        for pos in common_pos:
            sub_sem_list = word_sem[sub_word][pos]
            id2sem = id_to_sem[word][pos]
            # for i in range(len(id2sem)):
            for i in range(len(id2sem)):
                if i not in word_candidate[word][pos]:
                    word_candidate[word][pos][i] = []
                sem_set = id2sem[i]
                if sem_set in sub_sem_list:
                    # print(sub_word)
                    word_candidate[word][pos][i].append(sub_word)

if __name__ == '__main__':
    word_candidate = {}
    vocab_dict = build_vocab()
    word_sem,word_pos = gen_sem_dict(vocab_dict)
    id_dict = gen_id2sem(word_sem)
    vocab_list = list(vocab_dict.keys())
    count = 0
    num_words = len(vocab_list)
    for word in vocab_list:
        count += 1
        print(count,'/',num_words)
        add_word(word,vocab_list,word_candidate,id_dict,word_sem,word_pos)

    test_word = '把握'
    # for test_word in word_list:
    subsets = word_candidate[test_word]
    for pos,subwords in subsets.items():
        print("pos: ",pos)
        print("subwords: ",subwords)
        print()

    print(id_dict[test_word])
    with open("aux_files/word_candidate.pkl",'wb') as f:
        pickle.dump(word_candidate,f)
    with open("aux_files/senseid.pkl",'wb') as f:
        pickle.dump(id_dict,f)









