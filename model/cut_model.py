# @File  : n_gram_model.py
# @Author: Cuiqingyao
# @Date  : 18-11-29
# @Desc  : 
# @Contact: qingyaocui@gmail.com

import os
import re
from conf.settings import CLEAN_CORPUS, N_GRAM_MODEL, SMOOTH_METHOD, MAX_LEN_OF_WORD, WORD_DICT_FILE
from utils.util import *


class Ngram(object):

    def __init__(self, n=2, corpus=CLEAN_CORPUS,
                 model_file=N_GRAM_MODEL,
                 smooth_method=SMOOTH_METHOD):
        self.n = n
        self.corpus = corpus
        self.model_file = model_file
        self.smooth_method = smooth_method
        self.model = self.__build_model(n, corpus, model_file, smooth_method)

    def __str__(self):
        return 'model：%s-gram based on %s'%(self.n, self.corpus)

    def __build_model(self, n, corpus, model_file, smooth_method):
        if not os.path.exists(N_GRAM_MODEL):
            # 计算n_gram概率分布

            # P = {'w1|w2':propability ... }
            P = {}
            all_tokens = []
            with open(self.corpus, 'r', encoding='utf8') as f:
                for line in f:
                    pairs = re.split(r'\s+', line.strip())
                    tokens = []
                    for pair in pairs:
                        token, _ = pair.split('/')
                        tokens.append(token)
                all_tokens.append(tokens)
            for i in range(len(all_tokens)):
                index = 0
                p_word = all_tokens[i][index] + "|"+all_tokens[i][index + 1]
                c_wn_1_wn = 0
                c_wn_1 = 0
                pass





        else:
            return self.__load_model(model_file=N_GRAM_MODEL)

    def __load_model(self, model_file):
        '''
        加载概率模型
        :param model_file:
        :return:
        '''
        P_of_corpus = read_from_json(model_file)
        return P_of_corpus

    def __additive_smoothin(self):
        '''

        :return:
        '''
        pass


word_dict = read_from_json(WORD_DICT_FILE)

def forward_maximum_match(sentence):
    '''
    正向最大匹配
    :param sentence:输入句子
    :return:
    '''
    origin_sentence = sentence
    segment = []
    index = 0
    len_origin = len(origin_sentence)

    while origin_sentence != '':
        tail_index = len(origin_sentence)

        if tail_index >= MAX_LEN_OF_WORD:
            tail_index = MAX_LEN_OF_WORD

        look_like_word = origin_sentence[index:tail_index]
        while tail_index > 1:

            if look_like_word not in word_dict:
                tail_index -= 1
                look_like_word = look_like_word[index:tail_index]
            else:
                segment.append(look_like_word)
                origin_sentence = origin_sentence[len(look_like_word):len_origin]
                len_origin -= len(look_like_word)
                break
        else:
            tail_index -= 1
            segment.append(look_like_word)
            origin_sentence = origin_sentence[1:]
    print(segment)

    # return segment


def backward_maximun_match(sentence):
    '''
    逆向最大匹配
    :param sentence:输入句子
    :return:
    '''
    origin_sentence = sentence
    segment = []

    while origin_sentence != '':
        head_index = 0
        index = len(origin_sentence)
        if (index - head_index) >= MAX_LEN_OF_WORD:
            head_index = index - MAX_LEN_OF_WORD

        look_like_word = origin_sentence[head_index:index]
        while (index-head_index) > 1:

            if look_like_word not in word_dict:
                origin_head = head_index
                head_index += 1
                look_like_word = look_like_word[(head_index - origin_head):len(look_like_word)]
            else:
                segment.append(look_like_word)
                origin_sentence = origin_sentence[0:head_index]
                break
        else:
            head_index += 1
            segment.append(look_like_word)
            origin_sentence = origin_sentence[0:head_index-1]
    segment = segment[::-1]
    print(segment)

    # return segment



if __name__ == '__main__':
    print("backward_maximun_match:")
    backward_maximun_match('19980101-01-001-001迈向充满希望的新世纪——一九九八年新年讲话（附图片１张）')
    print('forward_maximun_match:')
    forward_maximum_match('19980101-01-001-001迈向充满希望的新世纪——一九九八年新年讲话（附图片１张）')