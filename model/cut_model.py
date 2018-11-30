# @File  : n_gram_model.py
# @Author: Cuiqingyao
# @Date  : 18-11-29
# @Desc  : 
# @Contact: qingyaocui@gmail.com

import os
import re
import json
from conf.settings import CLEAN_CORPUS_WITHOUT_POS, N_GRAM_MODEL, SMOOTH_METHOD, \
    MAX_LEN_OF_WORD, WORD_DICT_FILE, WORD_DICT_SIZE
from utils.util import *
from tqdm import tqdm


class Ngram(object):

    def __init__(self, n=2, corpus=CLEAN_CORPUS_WITHOUT_POS,
                 model_file=N_GRAM_MODEL,
                 smooth_method=SMOOTH_METHOD,
                 max_len=MAX_LEN_OF_WORD,
                 word_dict_file=WORD_DICT_FILE,
                 word_dict_size=WORD_DICT_SIZE):
        self.n = n
        self.corpus = corpus
        self.model_file = model_file
        self.smooth_method = smooth_method
        self.max_len = max_len
        self.word_dict_file = word_dict_file
        self.word_dict_size = word_dict_size
        self.model = self.__build_model()

    def __str__(self):
        return 'model：%s-gram based on %s'%(self.n, self.corpus)

    def __build_model(self):

        if not os.path.exists(self.model_file):
            print("构建2元语法模型...")
            # 计算n_gram概率分布
            P_n_gram = {}
            context, all_tokens = self.__load_training_data()
            word_dict = self.__load_word_dict()

            for i in tqdm(range(len(all_tokens))):
                index = 0
                while index < len(all_tokens[i]) - 1:

                    w2 = all_tokens[i][index + 1]
                    w1 = all_tokens[i][index]
                    p_word = w2 + '|' + w1

                    if p_word not in P_n_gram:
                        c_w1w2 = 0
                        c_w1 = word_dict[w1]['count']
                        for line in context:
                            c_w1w2 += len(re.findall(w1+' '+w2, line))

                        P_n_gram[p_word] = c_w1w2/c_w1
                        # print(p_word , P_n_gram[p_word])
                    index += 1
            write_to_json_min(P_n_gram, self.model_file)
            return P_n_gram
        else:
            print("模型已存在，正在加载...")
            return self.__load_model()

    def __load_training_data(self):
        context = []
        all_tokens = []
        with open(self.corpus, 'r', encoding='utf8') as f:
            for line in f:
                context.append(line.strip())
                tokens = re.split(r'\s+', line.strip())
                all_tokens.append(tokens)

        return context, all_tokens

    def __load_model(self):
        '''
        加载概率模型
        :param model_file:
        :return:
        '''
        P_n_gram = read_from_json(self.model_file)
        return P_n_gram

    def __load_word_dict(self):
        '''
        读取词典
        :param word_dict:
        :return:
        '''
        word_dict = read_from_json(self.word_dict_file)
        return word_dict

    def __additive_smoothin(self, c_w1w2, c_w2):
        '''
        加一平滑
        :return:
        '''
        c_w1w2 += 1
        c_w2 += self.word_dict_size
        return c_w1w2, c_w2


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
    # print("backward_maximun_match:")
    # backward_maximun_match('19980101-01-001-001迈向充满希望的新世纪——一九九八年新年讲话（附图片１张）')
    # print('forward_maximun_match:')
    # forward_maximum_match('19980101-01-001-001迈向充满希望的新世纪——一九九八年新年讲话（附图片１张）')
    # n_gram = Ngram()
    my_model = read_from_json(N_GRAM_MODEL)
    print(len(my_model))

    all_tokens = []
    with open(CLEAN_CORPUS_WITHOUT_POS, 'r', encoding='utf8') as f:
        for line in f:
            tokens = re.split(r'\s+', line.strip())
            all_tokens.append(tokens)
    s = 0
    for one_line in all_tokens:
        s += len(one_line) - 1

    print('s:', s)