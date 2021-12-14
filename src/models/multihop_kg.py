"""
Knowledge Graph
"""
import os
import sys
import csv
import torch
import string
import numpy as np
# from tqdm import tqdm
from nltk.tokenize import MWETokenizer
from nltk.corpus import stopwords
from nltk import pos_tag
from transformers import BertTokenizer, DistilBertTokenizer, DistilBertModel, DistilBertForSequenceClassification
sys.path.append(os.path.join(os.getcwd(), 'src')) # Move to the main module
from utils.loader import load_file
# from typing import List


class KnowledgeGraph:

    def __init__(self, params):
        if params['knowledge_graph'] == 'conceptnet':
            self.kg_path = params['conceptnet_path']
            self.kg_triples = params['conceptnet_triples_file']
            # self.kg_multi_words = params['conceptnet_multi_words_file']
            self.kg_multi_words = params['multi_words_file']

        multi_words = load_file(os.path.join(self.kg_path, self.kg_multi_words))
        self.MWETokenizer = MWETokenizer(multi_words)
        self.BertTokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.DistilBertTokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.DistilBertModel = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.stopwords = stopwords.words('english')

        self.nn_pos_tag = params['nn_pos_tag']
        self.n_hop = params['n_hop']
        self.max_entities = params['max_entities']
        self.excluded_predicates = params['excluded_predicates']
        self.predicate = params['predicate']
        self.split_multi_words = params['split_multi_words']
        self.max_num_tokens = params['max_num_tokens']
        self.cls_token = params['cls_token']
        self.sep_token = params['sep_token']
        self.pad_token = params['pad_token']

        self.lookup_table = self._create_lookup_table()
        # self.segment_vocab = list(self.lookup_table.keys()) + config.NEVER_SPLIT_TAG
        # self.special_tags = params['never_split_tag']


    def _create_lookup_table(self):
        """
        Returns
        -------
        dict: {'subject': {value1, value2}, ...}
        """
        lookup_table = {}
        print("Loading Knowledge Graph")
        with open(os.path.join(self.kg_path, self.kg_triples), 'r') as f:
            reader = csv.reader(f)
            for line in reader:
                subj, pred, obj = line
                if pred in self.excluded_predicates:
                    continue
                if self.predicate:
                    value = ' '.join([pred, obj])
                else:
                    value = obj
                if subj in lookup_table.keys():
                    lookup_table[subj].add(value)
                else:
                    lookup_table[subj] = set([value])
        return lookup_table


    def _remove_punctuations(self, x):
        return "".join(char for char in x if char not in string.punctuation)


    def _expand_branch(self, branch, soft_idx, depth_list, center_words):
        """
        Recursively expand branch and find n hop neighbours

        Parameters
        ----------
        branch: a list containing a center word and its neighbor words
        soft_idx: a list containing soft position indices of words
        depth_list: a list containing depth of each word
        center_words: a list containing tuples of (center word, pos, soft idx, depth)

        Returns
        -------
        branch, soft_idx, depth_list
        """
        while center_words != []:
            center_word, pos, center_soft_idx, depth = center_words.pop(0)
            branch += center_word
            soft_idx += [center_soft_idx + i for i in range(len(center_word))]
            depth_list += [depth for _ in range(len(center_word))]
            if center_word[-1] in self.stopwords or pos not in self.nn_pos_tag:
                continue
            neighbours = self.lookup_table.get(center_word[-1])
            if neighbours == None:
                continue
            else:
                # TODO: compute bert embeddings of neighbours and compute similarity
                # TODO: sort neighbours by score and keep top #max_entities neighbours
                n_neighbour = 0
                for neighbour in neighbours:
                    new_branch = neighbour.split()
                    if new_branch[-1] in branch:
                        continue
                    if depth < self.n_hop:
                        pos = pos_tag([new_branch[-1]])[0][-1]
                        center_words = [(new_branch, pos, center_soft_idx + len(center_word), depth + 1)] + center_words
                        n_neighbour += 1
                    if n_neighbour == self.max_entities:
                        break
                return self._expand_branch(branch, soft_idx, depth_list, center_words)
        return branch, soft_idx, depth_list


    def _compute_visible_matrix(self, tree_soft_idx, depth_list):
        tree_soft_idx = [0] + tree_soft_idx
        depth_list = [0] + depth_list
        sentence_len = len(depth_list)
        visible_matrix = np.zeros((sentence_len, sentence_len))
        for i in range(sentence_len):
            visible_matrix[i,i] = 1
            soft_idx = tree_soft_idx[i]
            my_child = True
            for j in range(i+1, sentence_len):
                if depth_list[i] == 0 and depth_list[j] == 0:
                    visible_matrix[i,j] = 1
                    visible_matrix[j,i] = 1
                    my_child = False
                else:
                    idx = tree_soft_idx[j]
                    if soft_idx < idx:
                        if not my_child:
                            continue
                        visible_matrix[i,j] = 1
                        visible_matrix[j,i] = 1
                    else:
                        my_child = False
                        break
        return visible_matrix


    def _split_multi_words(self, sentence_tree, tree_soft_idx, depth_list, visible_matrix):
        """
        e.g. "related_to" -> "related", "to"
        """
        new_sentence_tree = []
        new_tree_soft_idx = []
        new_depth_list = []
        for i, word in enumerate(sentence_tree):
            depth = depth_list[i]
            if i == 0:
                soft_idx = 1
            else:
                if depth == 0:
                    soft_idx = new_depth_list.count(0) + 1
                elif depth > depth_list[i - 1]:
                    soft_idx = new_tree_soft_idx[-1] + 1
                elif depth < depth_list[i - 1]:
                    try:
                        idx = len(new_depth_list) - new_depth_list[::-1].index(depth - 1)
                        soft_idx = new_tree_soft_idx[idx - 1] + 1
                    except:
                        raise Exception("can't roll back to previous center word")
                else:
                    if tree_soft_idx[i] > tree_soft_idx[i - 1]:  # same branch (predicate - object)
                        soft_idx = new_tree_soft_idx[-1] + 1
                    else:
                        try:
                            idx = len(new_depth_list) - new_depth_list[::-1].index(depth - 1)
                            soft_idx = new_tree_soft_idx[idx - 1] + 1
                        except:
                            raise Exception("can't roll back to previous center word")
            if '_' in word:
                new_words = word.split('_')
            else:
                new_words = [word]
            words_num = len(new_words)

            vm_size = visible_matrix.shape[0]
            l = len(new_sentence_tree) + 1
            for j in range(vm_size):
                vm = visible_matrix[j]
                row = np.concatenate([vm[:l], np.full(words_num, vm[l]), vm[(l + 1):]])#self.max_num_tokens]])
                if j == 0:
                    new_visible_matrix = np.zeros((len(row), len(row)))
                if j < l:
                    new_visible_matrix[j] = row
                elif j == l:
                    for k in range(words_num):
                        new_visible_matrix[j + k] = row
                else:
                    new_visible_matrix[j + words_num - 1] = row  # size
            visible_matrix = new_visible_matrix

            new_sentence_tree += new_words
            new_tree_soft_idx += list(range(soft_idx, soft_idx + words_num))
            new_depth_list += [depth for _ in range(words_num)]
        return new_sentence_tree, new_tree_soft_idx, new_depth_list, visible_matrix


    def _padding(self, token_ids, sentence_tree, tree_soft_idx, depth_list, visible_matrix):
        token_num = len(token_ids)
        sentence_tree = [self.cls_token] + sentence_tree
        tree_soft_idx = [0] + tree_soft_idx
        depth_list = [0] + depth_list
        if token_num <= self.max_num_tokens:
            pad = self.max_num_tokens - token_num # [CLS], [SEP]
            token_ids = token_ids + [0] * pad
            sentence_tree += [self.sep_token] + [self.pad_token] * pad
            tree_soft_idx += [self.max_num_tokens - 1] * (pad+1)
            depth_list += [0] * (pad+1)
            visible_matrix = np.pad(visible_matrix, ((0, pad+1), (0, pad+1)), mode='constant')
            mask = [1] * token_num + [0] * pad
        else:
            token_ids = token_ids[:self.max_num_tokens]
            sentence_tree = sentence_tree[:self.max_num_tokens]
            tree_soft_idx = tree_soft_idx[:self.max_num_tokens]
            depth_list = depth_list[:self.max_num_tokens]
            visible_matrix = visible_matrix[:self.max_num_tokens, :self.max_num_tokens]
            mask = [1] * self.max_num_tokens
        return token_ids, sentence_tree, tree_soft_idx, depth_list, visible_matrix, mask


    def add_knowledge(self, sent_batch):
        """
        Append n hop neighbours to the original sentence

        Parameters
        ----------
        sent_batch: list of sentences

        Returns
        -------
        sent_tree_batch, soft_idx_batch, visible_matrix_batch, depth_batch
        """
        sent_tree_batch = []
        token_id_batch = []
        soft_idx_batch = []
        visible_matrix_batch = []
        depth_batch = []
        mask_batch = []

        for sentence in sent_batch:
            sentence = self._remove_punctuations(sentence)
            # sent = self.DistilBertTokenizer(sentence.lower())
            # self.DistilBertModel(sent['input_ids'])
            sent_split = self.BertTokenizer.tokenize(sentence.lower())
            sent_split = self.MWETokenizer.tokenize(sent_split)
            sent_split = pos_tag(sent_split)

            sentence_tree = []  # a list of words in a sentence with n hop neighbour words
            tree_soft_idx = []  # K-BERT soft-position indices
            depth_list = []
            for i, (word_token, pos) in enumerate(sent_split):
                # TODO: compute bert embedding of word_token and pass it to _expand_branch
                depth = 0
                soft_idx = i + 1
                center_words = [([word_token], pos, soft_idx, depth)]

                sentence_tree, tree_soft_idx, depth_list = self._expand_branch(sentence_tree, tree_soft_idx, depth_list,
                                                                               center_words)

            # visible matrix
            visible_matrix = self._compute_visible_matrix(tree_soft_idx, depth_list)

            if self.split_multi_words:
                sentence_tree, tree_soft_idx, depth_list, visible_matrix = \
                    self._split_multi_words(sentence_tree, tree_soft_idx, depth_list, visible_matrix)

            token_ids = self.BertTokenizer.encode(sentence_tree)
            token_ids, sentence_tree, tree_soft_idx, depth_list, visible_matrix, mask = \
                self._padding(token_ids, sentence_tree, tree_soft_idx, depth_list, visible_matrix)

            sent_tree_batch.append(sentence_tree)
            token_id_batch.append(token_ids)
            soft_idx_batch.append(tree_soft_idx)
            visible_matrix_batch.append(visible_matrix)
            depth_batch.append(depth_list)
            mask_batch.append(mask)

        token_id_batch = torch.LongTensor(token_id_batch)
        soft_idx_batch = torch.LongTensor(soft_idx_batch)
        visible_matrix_batch = torch.LongTensor(visible_matrix_batch)
        depth_batch = torch.LongTensor(depth_batch)
        mask_batch = torch.LongTensor(mask_batch)

        return token_id_batch, mask_batch, soft_idx_batch, visible_matrix_batch, depth_batch, sent_tree_batch
