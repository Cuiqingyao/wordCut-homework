# @File  : settings.py
# @Author: Cuiqingyao
# @Date  : 18-11-29
# @Desc  : 
# @Contact: qingyaocui@gmail.com
import os

CORPUS = '/home/cuiqy/wordCut-homework/data/199801.txt'

CLEAN_CORPUS = os.path.join(os.path.dirname(CORPUS), os.path.basename(CORPUS).split('.')[0] + '_clean.txt')

WORD_DICT_FILE = '/home/cuiqy/wordCut-homework/data/word_dict.json'

MAX_LEN_OF_WORD = 26

N_GRAM_MODEL = '/home/cuiqy/wordCut-homework/data/n_gram.json'

SMOOTH_METHOD = 'additive_smoothing' # GT Katz


