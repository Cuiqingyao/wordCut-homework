# @File  : data_helper.py
# @Author: Cuiqingyao
# @Date  : 18-11-29
# @Desc  : 
# @Contact: qingyaocui@gmail.com

import re
from utils.util import write_to_json, write_to_json_min, clean_data, read_from_json
from conf.settings import CORPUS, CLEAN_CORPUS, WORD_DICT_FILE


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
if __name__ == '__main__':
    max_len = max_len_of_word()
    print(max_len)