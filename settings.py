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

# 语料相关统计
WORD_TYPES = 52543
MAX_LEN_OF_WORD = 5
TOKENS = 1017982
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
N_GRAM_MODEL = os.path.join(BASE_DIR, 'data', 'n_gramadditive_smoothing.json')

################    N-gram    ################

# 最终输出结果
RESULT_FILE = os.path.join(BASE_DIR, '2018140546.txt')