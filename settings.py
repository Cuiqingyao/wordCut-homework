# @File  : settings.py
# @Author: Cuiqingyao
# @Date  : 18-11-29
# @Desc  : 
# @Contact: qingyaocui@gmail.com
import os

CORPUS = './data/199801.txt'

CLEAN_CORPUS = os.path.join(os.path.dirname(CORPUS), os.path.basename(CORPUS).split('.')[0] + '_clean.txt')

WORD_DICT_FILE = './data/word_dict.json'

