# @File  : settings.py
# @Author: Cuiqingyao
# @Date  : 18-11-29
# @Desc  : 
# @Contact: qingyaocui@gmail.com
import os

CORPUS = '/home/cuiqy/wordCut-homework/data/199801.txt'

CLEAN_CORPUS = os.path.join(os.path.dirname(CORPUS), os.path.basename(CORPUS).split('.')[0] + '_clean.txt')
CLEAN_CORPUS_WITHOUT_POS = os.path.join(os.path.dirname(CORPUS), os.path.basename(CORPUS).split('.')[0] + '_clean_without_pos.txt')

WORD_DICT_FILE = '/home/cuiqy/wordCut-homework/data/word_dict.json'
WORD_DICT_SIZE = 74794
MAX_LEN_OF_WORD = 26

# None 不加平滑
# additive_smoothing 加1平滑
# GT good-turing
# katz Katz
SMOOTH_METHOD = 'additive_smoothing'
N_GRAM_MODEL = '/home/cuiqy/wordCut-homework/data/n_gram%s.json'%(SMOOTH_METHOD)




