import tensorflow as tf
import os
import numpy as np
import copy
import pickle
import math
import torch
from transformers import BertForMaskedLM, BertTokenizer
from sklearn.cluster import MeanShift, estimate_bandwidth 
import tensorflow_hub as hub
import pandas as pd
import OpenHowNet



def match_set(set_list, target_set):
    for i in range(len(set_list)):
        if target_set == set_list[i]:
            return i
    return -1

class log_module():
    def __init__(self):
        self.word_dict = {}
    
    def add_word(self,word,pos,select_sense, real_sense):
        if word not in self.word_dict:
            self.word_dict[word] = []
        data = {}
        data['word'] = word
        data['pos'] = pos
        data['select_sense'] = select_sense
        data['real_sense'] = real_sense
        self.word_dict[word].append(data)


class Dictionary():
    def __init__(self):
        self.word_dict = {}
        # count = 0
        with open("sgns.merge.word",'r',encoding = 'utf-8') as f:
            next(f)
            for line in f:
                items = line.split()
                # try:
                word = ''.join(items[:-300])
                vectors = np.array([float(x) for x in items[-300:]])
                # except:
                    # print(word)
                    # print(items[:5])
                self.word_dict[word] = vectors
        keys = list(self.word_dict.keys())[:20]
        # print(self.word_dict.keys())
        for key in keys:
            print(key,self.word_dict[key][:10])

    def __call__(self,word):
        if word in self.word_dict:
            return self.word_dict[word]
        else:
            return None

class dense_filter():
    def __init__(self):
        with open("vector_dict.pkl",'rb') as f:
            self.word_dict = pickle.load(f)
        # self.word_dict = Dictionary()
        self.correct = 0
        self.word_count = 0    
        self.sense_num = 0
        self.rand_count = 0
        self.load_dict()
        
    def load_all_words_data(self):
        with open("aux_files/anotation.pkl",'rb') as f:
            all_data = pickle.load(f)
        return all_data
    
    def load_dict(self):
        with open("aux_files/senseid.pkl",'rb') as f:
            self.word_sense_id_sem = pickle.load(f)
        with open("aux_files/word_candidate.pkl",'rb') as f:
            self.word_candidate = pickle.load(f)
        # with open("word_sem")


    def test_model(self):

        logger = log_module()

        all_data = []
        noun_count = 0
        noun_correct = 0
        verb_count = 0
        verb_correct = 0
        with open("dataset.txt",'r',encoding = 'utf-8') as f:
            for line in f:
                data = eval(line.strip())
                all_data.append(data)
        pos_dict = {
                'n':'noun',
                'v':'verb', 
                'vn':'verb',
                'vd':'verb',
                'ns':'noun',
                'a':'adj', 
                'an':'adj',
                'd':'adv', 
                'ad':'adv',
            }
        check_pos_list = ['noun', 'verb', 'adj', 'adv']
        for i in range(len(all_data)):
            item = all_data[i]
            # print(curr_data)
            if len(item) == 0:
                continue
            context = item['context']
            pos_list = item['part-of-speech']
            target_word = item['target_word']
            target_position = item['target_position']
            target_word_pos = item['target_word_pos']
            sense_set = item['sense']
            print(item['target_position'],target_word)
            if '?' in sense_set:
                sense_set.remove('?')            
            if target_word_pos not in pos_dict:
                print("pos not in valid list: ",target_word_pos)
                continue
            transformed_pos = pos_dict[target_word_pos]
            word_sentence = context.copy()
            word_sentence[target_position] = target_word

            context_pos = []
            for item in pos_list:
                if item not in pos_dict:
                    context_pos.append(None)
                else:
                    context_pos.append(pos_dict[item])

            if target_word not in self.word_candidate:
                continue
            if transformed_pos not in self.word_candidate[target_word]:
                continue

            sub_dict = self.word_candidate[target_word][transformed_pos]
            # print(sentence)
            print(target_word,transformed_pos)
            print(target_position)
            # for idx, subwords in sub_dict.items():
            #     print('\t',idx, subwords)
            self.sense_num += len(sub_dict)
            sense_dict = self.word_sense_id_sem[target_word][transformed_pos]
            # target_index,prob_list = self.select_sense(context_words,target_word,sense_dict)
            target_index,prob_list = self.select_sense(word_sentence,context_pos,target_word,sub_dict)
            print(target_index)
            select_sem_set = self.word_sense_id_sem[target_word][transformed_pos][target_index]

            print()
            print("select sense: ", select_sem_set," prob: ",prob_list[target_index])
            print("real sense:", sense_set)

            real_sense_idx = match_set(self.word_sense_id_sem[target_word][transformed_pos],sense_set)
            if real_sense_idx == -1:
                continue
            logger.add_word(target_word,transformed_pos,target_index, real_sense_idx)


            if select_sem_set == sense_set:
                print("!")
                self.correct += 1
                if transformed_pos == 'noun':
                    noun_correct += 1
                elif transformed_pos == 'verb':
                    verb_correct += 1
                else:
                    print("error: ",transformed_pos)
                    pause = input("?")
            self.word_count += 1
            if transformed_pos == 'noun':
                noun_count += 1
            elif transformed_pos == 'verb':
                verb_count += 1
            print()
            print('-'*60)
            print()
            # pause = input("continue? ")
        print(self.correct,self.word_count)
        print(self.word_count,self.sense_num)
        print(noun_correct,noun_count)
        print(verb_correct,verb_count)
        with open("dense_log.pkl",'wb') as f:
            pickle.dump(logger,f)

    def select_sense(self,word_sentence,word_pos_list,orig_word,sub_dict):
        # if filter_dict == None:
        sim_score_list = []
        sentence_vector = []
        # for x in word_sentence:
        for i in range(len(word_sentence)):
            x = word_sentence[i]
            pos = word_pos_list[i]
            if pos is None:
                continue
            vec = self.word_dict(x)
            # print(vec.shape)
            if vec is not None:
                sentence_vector.append(vec)
        if len(sentence_vector) == 0:
            print(word_sentence)
            print(word_pos_list)
            
        sentence_vector = np.stack(sentence_vector,axis = 0).astype(np.float32)
        sentence_vector = np.mean(sentence_vector,axis = 0)
        assert sentence_vector.shape[0] == 300
        for i in range(len(sub_dict)):
            subwords = sub_dict[i]
            unique_list = []
            for subword in subwords:
                if subword == orig_word:
                    continue
                else:
                    unique_list.append(subword)
            final_list = unique_list
            
            score = self.cal_synset_score(sentence_vector,final_list)
            sim_score_list.append(score)
        target_index = np.argmax(sim_score_list)
        return target_index,sim_score_list
    
    def cal_synset_score(self,sentence_vector,final_list):
        if len(final_list) == 0:
            return -1
        synset_vec = []
        for x in final_list:
            vec = self.word_dict(x)
            if vec is not None:
                synset_vec.append(vec)
        synset_vec = np.stack(synset_vec,axis = 0)
        synset_vec = np.mean(synset_vec,axis = 0)
        score = np.dot(sentence_vector,synset_vec)/np.sqrt((np.sum(np.square(sentence_vector)) * np.sum(np.square(sentence_vector))))
        print(score)
        return score

class bert_filter():
    def __init__(self,model_type = 'bert-base-chinese'):
        self.bert_model = BertForMaskedLM.from_pretrained(model_type).to("cuda")
        self.bert_model.eval()
        self.tokenizer = BertTokenizer.from_pretrained(model_type)
        self.correct = 0
        self.word_count = 0    
        self.sense_num = 0
        self.rand_count = 0
        self.load_dict()
        
    
    def load_dict(self):
        with open("aux_files/senseid.pkl",'rb') as f:
            self.word_sense_id_sem = pickle.load(f)
        with open("aux_files/word_candidate.pkl",'rb') as f:
            self.word_candidate = pickle.load(f)

    def cal_prob_batch(self,sentence,orig_word,sub_word_list,position):
        mask_char = ['[MASK]' for _ in range(len(sub_word_list[0]))] 
        copy_sentence = sentence[:position] + mask_char + sentence[position + len(orig_word):]
        copy_sentence = ['[CLS]'] + copy_sentence
        copy_sentence += ['[SEP]']
        text = ' '.join(copy_sentence) 
        bert_tokens = self.tokenizer.tokenize(text)
        # print(sentence)
        # print(bert_tokens)
        id_list = self.tokenizer.convert_tokens_to_ids(bert_tokens)
        input_ids = torch.tensor([id_list]).to("cuda")
        outputs = self.bert_model(input_ids,masked_lm_labels = input_ids)
        pre_scores = outputs[1].detach().cpu().numpy()[0]
        all_probs = []

        char_sub = [list(sub_word) for sub_word in sub_word_list]
        subword_ids = [self.tokenizer.convert_tokens_to_ids(x) for x in char_sub]

        for idx in range(len(subword_ids)):
            temp_list = []
            for pos in range(len(sub_word_list[idx])):
                word_id = subword_ids[idx][pos]
                temp_list.append(pre_scores[pos+position+1][word_id])
            all_probs.append(np.mean(temp_list))
        # print(all_probs)
        # pause = input("?")
        return all_probs

    def predict_synset_prob_batch(self,sentence,position,orig_word,sub_list):
        new_sen = sentence.copy()
        prob_list = []
        count = 0
        for idx in range(position,position + len(orig_word)):
            new_sen[idx] = '[MASK]'
        
        sub_length = [len(x) for x in sub_list]
        index_dict = {}
        for idx in range(len(sub_length)):
            length = sub_length[idx]
            if length not in index_dict:
                index_dict[length] = [idx]
            else:
                index_dict[length].append(idx)
        prob_list = []
        for length, index_list in index_dict.items():
            curr_subwords = [sub_list[x] for x in index_list]
            curr_probs = self.cal_prob_batch(sentence, orig_word,curr_subwords,position)
            prob_list += curr_probs
        return prob_list

    def select_sense(self,sentence, positions, orig_word,sub_dict):
        avg_prob_list = []
        for i in range(len(sub_dict)):
            subwords = sub_dict[i]
            unique_list = []
            for subword in subwords:
                if subword == orig_word:
                    continue
                else:
                    unique_list.append(subword)

            final_list = unique_list

            prob_list = self.predict_synset_prob_batch(sentence,positions,orig_word,final_list)
            if len(prob_list) >= 1:
                avg_prob_list.append(np.mean(prob_list))
            else:
                avg_prob_list.append(-10)
        target_index = np.argmax(avg_prob_list)
        print("selection: %d"%(target_index))
        print()
        return target_index,avg_prob_list

def get_annotation(word):
    sememes = hownet_dict.get_sememes_by_word(word,structured = False,lang = 'zh',merge = True)
    if len(sememes) == 1:
        sememe_list = []
        for sem in sememes:
            sememe_list.append(sem)
        return sememe_list[0]
    else:
        return None

class cip15_filter():
    def __init__(self,model_path = 'cip15_model/',vec_dim = 300,window_size = 8):
        self.vec_dim = vec_dim
        self.model_path = model_path
        self.window_size = window_size
        self.word_vec_dict, self.sem_vec_dict = self.load_vector()
        self.correct = 0
        self.word_count = 0    
        self.sense_num = 0
        self.rand_count = 0
        self.hownet_dict = OpenHowNet.HowNetDict()
        self.load_dict()
    
    def load_vector(self):
        with open(self.model_path + "word_vec.pkl",'rb') as f:
            word_vec = pickle.load(f)
        with open(self.model_path + "sem_vec.pkl",'rb') as f:
            sem_vec = pickle.load(f)
        return word_vec,sem_vec

    def get_annotation(self,word):
        try:
            sememes = self.hownet_dict.get_sememes_by_word(word,structured = False,lang = 'zh',merge = True)
        except:
            return None
        if len(sememes) == 1:
            sememe_list = []
            for sem in sememes:
                sememe_list.append(sem)
            return sememe_list[0]
        else:
            return None


    def load_all_words_data(self):
        with open("aux_files/anotation.pkl",'rb') as f:
            all_data = pickle.load(f)
        return all_data
    
    def load_dict(self):
        with open("aux_files/senseid.pkl",'rb') as f:
            self.word_sense_id_sem = pickle.load(f)
        with open("aux_files/word_candidate.pkl",'rb') as f:
            self.word_candidate = pickle.load(f)
        # with open("word_sem")

    def test_model(self):

        logger = log_module()

        all_data = []
        noun_count = 0
        noun_correct = 0
        verb_count = 0
        verb_correct = 0
        with open("dataset.txt",'r',encoding = 'utf-8') as f:
            for line in f:
                data = eval(line.strip())
                all_data.append(data)
        pos_dict = {
                'n':'noun',
                'v':'verb', 
                'vn':'verb',
                'vd':'verb',
                'ns':'noun',
                'a':'adj', 
                'an':'adj',
                'd':'adv', 
                'ad':'adv',
            }
        check_pos_list = ['noun', 'verb', 'adj', 'adv']
        for i in range(len(all_data)):
            item = all_data[i]
            # print(curr_data)
            if len(item) == 0:
                continue
            context = item['context']
            pos_list = item['part-of-speech']
            target_word = item['target_word']
            target_position = item['target_position']
            target_word_pos = item['target_word_pos']
            sense_set = item['sense']
            if '?' in sense_set:
                sense_set.remove('?')
            print(item['target_position'],target_word)
            context_words = []
            for idx in range(max([0,target_position - self.window_size]), min([len(context),target_position + self.window_size + 1])):
                if idx == target_position:
                    continue
                context_words.append(context[idx])
            if target_word_pos not in pos_dict:
                    print("pos not in valid list: ",target_word_pos)
                    continue
            transformed_pos = pos_dict[target_word_pos]

            if target_word not in self.word_candidate:
                continue
            if transformed_pos not in self.word_candidate[target_word]:
                continue

            sub_dict = self.word_candidate[target_word][transformed_pos]
            # print(sentence)
            print(target_word,transformed_pos)
            print(target_position)
            # for idx, subwords in sub_dict.items():
            #     print('\t',idx, subwords)
            self.sense_num += len(sub_dict)
            sense_dict = self.word_sense_id_sem[target_word][transformed_pos]
            target_index,prob_list = self.select_sense(context_words,target_word,sense_dict)
            print(target_index)
            select_sem_set = self.word_sense_id_sem[target_word][transformed_pos][target_index]

            real_sense_idx = match_set(self.word_sense_id_sem[target_word][transformed_pos],sense_set)
            if real_sense_idx == -1:
                continue
            logger.add_word(target_word,transformed_pos,target_index, real_sense_idx)


            print()
            print("select sense: ", select_sem_set," prob: ",prob_list[target_index])
            print("real sense:", sense_set)

            if select_sem_set == sense_set:
                print("!")
                self.correct += 1
                if transformed_pos == 'noun':
                    noun_correct += 1
                elif transformed_pos == 'verb':
                    verb_correct += 1
                else:
                    print("error: ",transformed_pos)
                    pause = input("?")
            self.word_count += 1
            if transformed_pos == 'noun':
                noun_count += 1
            elif transformed_pos == 'verb':
                verb_count += 1
            print()
            print('-'*60)
            print()
            # pause = input("continue? ")
        print(self.correct,self.word_count)
        print(self.word_count,self.sense_num)
        print(noun_correct,noun_count)
        print(verb_correct,verb_count)

        with open("cip15_log.pkl",'wb') as f:
            pickle.dump(logger,f)
    

    def select_sense(self,context_words,target_word,sense_dict):
        context_embedding = []
        for word in context_words:
            if word in self.word_vec_dict:
                context_embedding.append(self.word_vec_dict[word])
            else:
                annotation = self.get_annotation(word)
                if annotation == None:
                    print("word ",word,"not in the dict!")
                elif annotation in self.sem_vec_dict:
                    print(word,"with only one sememe: ",annotation)
                    context_embedding.append(self.sem_vec_dict[annotation])
        if len(context_embedding) == 0:
            return -1
        context_embedding = np.mean(np.stack(context_embedding,axis = 0),axis = 0)
        sim_score_list = []
        for i in range(len(sense_dict)):
            sense_set = sense_dict[i]
            sense_embeddings = []
            for sem in sense_set:
                if sem not in self.sem_vec_dict:
                    continue
                sense_embeddings.append(self.sem_vec_dict[sem])
            if len(sense_embeddings) == 0:
                sim_score_list.append(-1)
            else:
                sense_embeddings = np.mean(np.stack(sense_embeddings,axis = 0),axis = 0)
                sim_score = self.cos_sim(context_embedding,sense_embeddings)
                sim_score_list.append(sim_score)
        target_index = np.argmax(sim_score_list)
        return target_index, sim_score_list

    def cos_sim(self,vec1,vec2):
        return np.dot(vec1,vec2)/np.sqrt(np.sum(np.square(vec1)) * np.sum(np.square(vec2)))

class rand_filter():
    def __init__(self,model_path = 'cip15_model/',vec_dim = 300,window_size = 8):
        self.vec_dim = vec_dim
        self.model_path = model_path
        self.window_size = window_size
        self.correct = 0
        self.word_count = 0    
        self.sense_num = 0
        self.rand_count = 0
        self.hownet_dict = OpenHowNet.HowNetDict()
        self.load_dict()
    

    def load_all_words_data(self):
        with open("aux_files/anotation.pkl",'rb') as f:
            all_data = pickle.load(f)
        return all_data
    
    def load_dict(self):
        with open("aux_files/senseid.pkl",'rb') as f:
            self.word_sense_id_sem = pickle.load(f)
        with open("aux_files/word_candidate.pkl",'rb') as f:
            self.word_candidate = pickle.load(f)
        # with open("word_sem")

    def test_model(self):

        logger = log_module()

        all_data = []
        noun_count = 0
        noun_correct = 0
        verb_count = 0
        verb_correct = 0
        with open("dataset.txt",'r',encoding = 'utf-8') as f:
            for line in f:
                data = eval(line.strip())
                all_data.append(data)
        pos_dict = {
                'n':'noun',
                'v':'verb', 
                'vn':'verb',
                'vd':'verb',
                'ns':'noun',
                'a':'adj', 
                'an':'adj',
                'd':'adv', 
                'ad':'adv',
            }
        check_pos_list = ['noun', 'verb', 'adj', 'adv']
        for i in range(len(all_data)):
            item = all_data[i]
            # print(curr_data)
            if len(item) == 0:
                continue
            context = item['context']
            pos_list = item['part-of-speech']
            target_word = item['target_word']
            target_position = item['target_position']
            target_word_pos = item['target_word_pos']
            sense_set = item['sense']
            print(item['target_position'],target_word)
            if '?' in sense_set:
                sense_set.remove('?')
            context_words = []
            for idx in range(max([0,target_position - self.window_size]), min([len(context),target_position + self.window_size + 1])):
                if idx == target_position:
                    continue
                context_words.append(context[idx])
            if target_word_pos not in pos_dict:
                    print("pos not in valid list: ",target_word_pos)
                    continue
            transformed_pos = pos_dict[target_word_pos]

            if target_word not in self.word_candidate:
                continue
            if transformed_pos not in self.word_candidate[target_word]:
                continue

            sub_dict = self.word_candidate[target_word][transformed_pos]
            # print(sentence)
            print(target_word,transformed_pos)
            print(target_position)
            # for idx, subwords in sub_dict.items():
            #     print('\t',idx, subwords)
            self.sense_num += len(sub_dict)
            sense_dict = self.word_sense_id_sem[target_word][transformed_pos]
            target_index = self.select_sense(sense_dict)
            print(target_index)
            select_sem_set = self.word_sense_id_sem[target_word][transformed_pos][target_index]

            print()
            print("select sense: ", select_sem_set)
            print("real sense:", sense_set)

            real_sense_idx = match_set(self.word_sense_id_sem[target_word][transformed_pos],sense_set)
            if real_sense_idx == -1:
                continue
            logger.add_word(target_word,transformed_pos,target_index, real_sense_idx)

            if select_sem_set == sense_set:
                print("!")
                self.correct += 1
                if transformed_pos == 'noun':
                    noun_correct += 1
                elif transformed_pos == 'verb':
                    verb_correct += 1
                else:
                    print("error: ",transformed_pos)
                    pause = input("?")
            self.word_count += 1
            if transformed_pos == 'noun':
                noun_count += 1
            elif transformed_pos == 'verb':
                verb_count += 1
            print()
            print('-'*60)
            print()
            # pause = input("continue? ")
        print(self.correct,self.word_count)
        print(self.word_count,self.sense_num)
        print(noun_correct,noun_count)
        print(verb_correct,verb_count)

        with open("rand_log.pkl",'wb') as f:
            pickle.dump(logger,f)

    def select_sense(self,sense_dict):
        return np.random.randint(len(sense_dict))

