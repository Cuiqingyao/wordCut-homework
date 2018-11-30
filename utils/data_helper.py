# @File  : data_helper.py
# @Author: Cuiqingyao
# @Date  : 18-11-29
# @Desc  : 
# @Contact: qingyaocui@gmail.com

import re
import os
from utils.util import write_to_json, write_to_json_min, read_from_json
from conf.settings import CORPUS, CLEAN_CORPUS, CLEAN_CORPUS_WITHOUT_POS, WORD_DICT_FILE


def build_dict(corpus=CORPUS, word_dict_file=WORD_DICT_FILE):
    '''
    构建语料库词典
    :param corpus:
    :param word_dict_file:
    :return:
    '''
    try:
        clean_data(corpus, clean_corpus=CLEAN_CORPUS)
    except Exception as e:
        print(e)
        exit()
    word_dict = {}
    with open(CLEAN_CORPUS, 'r', encoding='utf8') as f:
        for line in f:
            pairs = re.split(r'\s+', line.strip())

            for pair in pairs:

                token, pos = pair.split('/')
                if token == '':
                    continue
                if token not in word_dict:
                    word_dict[token] = {}
                    word_dict[token]['id'] = len(word_dict)
                    word_dict[token]['pos'] = []
                    word_dict[token]['pos'].append(pos)
                    word_dict[token]['count'] = 1
                else:
                    word_dict[token]['count'] += 1
                    if pos not in word_dict[token]['pos']:
                        word_dict[token]['pos'].append(pos)
    write_to_json(word_dict, word_dict_file)
    write_to_json_min(word_dict, './data/word_dict_min.json')

def max_len_of_word(word_dict_file=WORD_DICT_FILE):
    word_dict = read_from_json(word_dict_file)
    max_len = 0
    for word in word_dict:
        if max_len < len(word):
            max_len = len(word)
    return max_len

def word_dict_size(word_dict_file=WORD_DICT_FILE):
    word_dict = read_from_json(word_dict_file)
    return len(word_dict)

def clean_data(corpus, clean_corpus):
    '''
    清洗数据，除去空行，并为每句话增加起始和终止标记
    :param corpus:
    :param clean_corpus:
    :return:
    '''
    if os.path.exists(corpus):
        if not os.path.exists(clean_corpus):
            with open(corpus, 'r', encoding='utf8') as f_in:
                with open(clean_corpus, 'w', encoding='utf8') as f_out:
                    for line in f_in:
                        if len(line.split()) != 0:
                            f_out.write('<B>/BEG  ' + line.strip() + '  <E>/END\n')
    else:
        raise Exception('原始数据路径不存在！')

def clean_data_without_pos(corpus, clean_corpus):
    '''
    清洗数据,去除词性标注
    :param corpus:
    :param clean_corpus:
    :return:
    '''

    if os.path.exists(corpus):
        if not os.path.exists(clean_corpus):
            with open(corpus, 'r', encoding='utf8') as f_in:
                with open(clean_corpus, 'w', encoding='utf8') as f_out:
                    for line in f_in:
                        pairs = re.split(r'\s+', line.strip())
                        context = ''
                        for pair in pairs:
                            token, _ = pair.split('/')
                            context += token + ' '
                        f_out.write(context + '\n')

    else:
        raise Exception('原始数据路径不存在！')

if __name__ == '__main__':
    # clean_data_without_pos(CLEAN_CORPUS, CLEAN_CORPUS_WITHOUT_POS)
    print(word_dict_size())