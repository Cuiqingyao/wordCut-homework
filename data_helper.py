# @File  : data_helper.py
# @Author: Cuiqingyao
# @Date  : 18-11-29
# @Desc  : 
# @Contact: qingyaocui@gmail.com

import re
from util import write_to_json, write_to_json_min, clean_data, read_from_json
from settings import CORPUS, CLEAN_CORPUS, WORD_DICT_FILE
from tqdm import tqdm

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


if __name__ == '__main__':
    build_dict()
    # word_content = read_from_json(WORD_CONTENT_FILE)
    # word_index = read_from_json(WORD_INDEX_FILE)
    # print(len(word_content))
    # print(word_content[word_index['19980106-01-005-007']])
