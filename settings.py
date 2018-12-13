# @File  : settings.py
# @Author: Cuiqingyao
# @Date  : 18-11-29
# @Desc  : 
# @Contact: qingyaocui@gmail.com
import os

# project root dir
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

################    Corpus    ################

# 原始语料路径
ORIGIN_CORPUS = os.path.join(BASE_DIR, 'data', 'origin_data', '199801.txt')

# 训练数据路径
TRAINING_FILE = os.path.join(BASE_DIR, 'data', 'training.txt')

# 测试数据路径
TEST_FILE = os.path.join(BASE_DIR, 'data', 'test.txt')

# 分词答案路径
ANSWERS = os.path.join(BASE_DIR, 'data', 'answers.txt')

# 词典路径
WORD_DICT_FILE =  os.path.join(BASE_DIR, 'data', 'word_dict.json')

# 训练数据和测试数据大小占比
TRAINING = 0.9
TEST = 1 - TRAINING

# 最大词长, 影响FMM 和 BMM
MAX_LEN_OF_WORD = 5

################    Corpus    ################


################    Hidden Markov Model    ################

# 初始概率
INIT_STATUS = os.path.join(BASE_DIR, 'data', 'hmm', 'init_status_prob.txt')

# 发射概率矩阵
EMISSION_MATRIX = os.path.join(BASE_DIR, 'data', 'hmm', 'emit_matrix .json')

# 转移概率矩阵
TRANS_MATRIX = os.path.join(BASE_DIR, 'data', 'hmm', 'trans_matrix.txt')

################    Hidden Markov Model    ################


################    N-gram    ################

# n-gram模型统计数据
BIGRAM_PROB_TAB_EVAL = os.path.join(BASE_DIR, 'data', 'n_gram','bigram_prob_tab_eval.json')

# n-gram模型统计数据
TRAINING_FILE_FOR_BIGRAM = os.path.join(BASE_DIR, 'data', 'n_gram','199801-bigram.txt')

# 基于全语料的词典
WORD_DICT_FILE_FOR_BIGRAM =  os.path.join(BASE_DIR, 'data', 'n_gram', 'word_dict_bigram.json')

# 基于全语料的词典
BIGRAM_PROB_TAB =  os.path.join(BASE_DIR, 'data', 'n_gram', 'bigram_prob_tab.json')

# n-gram 是否用于评估
IS_EVALUATION = True

################    N-gram    ################

# 目标分词文件
TARGET_FILE = os.path.join(BASE_DIR, 'test.txt')

# 最终输出结果
RESULT_FILE = os.path.join(BASE_DIR, '2018140546.txt')