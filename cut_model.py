# @File  : n_gram_model.py
# @Author: Cuiqingyao
# @Date  : 18-11-29
# @Desc  : 
# @Contact: qingyaocui@gmail.com

import numpy as np
import math
from settings import *
from utils.util import *
from tqdm import tqdm
from collections import defaultdict


class Bigram(object):

    def __init__(self):
        self.is_evaluation = IS_EVALUATION
        if self.is_evaluation:
            self.corpus = TRAINING_FILE
            self.word_dict_file = WORD_DICT_FILE
            self.bigram_file = BIGRAM_PROB_TAB_EVAL
        else:
            self.corpus = TRAINING_FILE_FOR_BIGRAM
            self.word_dict_file = WORD_DICT_FILE_FOR_BIGRAM
            self.bigram_file = BIGRAM_PROB_TAB
        self.word_dict = self.__load_word_dict()
        self.bigram = self.__build_model()

    def __str__(self):
        return 'Bigram Model'

    def __construct_segment(self, sentence, max_len=5):
        '''
        递归构造所有可能的切分方式
        :return:可能的切分组合
        '''
        # 切分组合  [ [切分方案1], [切分方案2], ... ]

        splits = []

        # 递归结束条件
        if len(sentence) < 2:
            return [[sentence]]

        scope_max = max_len if max_len < len(sentence) else len(sentence)
        # 保存当前二切分结果
        current_groups = []
        single_cut = True
        for i in range(1, scope_max)[::-1]:
            # 词典存在子词串，二分切分
            if sentence[:i] in self.word_dict and i != 1:
                current_groups.append([sentence[:i], sentence[i:]])
                single_cut = False
        if single_cut or sentence[1:3] in self.word_dict:
            current_groups.append([sentence[:1], sentence[1:]])

        if len(sentence) <= 3:
            current_groups.append([sentence])
        # 对每一个切分，递归组合
        for one_group in current_groups:
            if len(one_group) == 1:
                splits.append(one_group)
                continue
            for child_group in self.__construct_segment(one_group[1]):
                child_group.insert(0, one_group[0])
                splits.append(child_group)

        return splits


    def __build_model(self):

        if not os.path.exists(self.bigram_file):
            print("构建bi-gram...")
            # 计算n_gram概率分布
            count_bigram = defaultdict(dict)

            all_tokens = self.__load_training_data()

            for i in tqdm(range(len(all_tokens))):
                index = 0
                while index < len(all_tokens[i]) - 1:

                    cur_word = all_tokens[i][index + 1]
                    front_word = all_tokens[i][index]
                    if front_word not in count_bigram:
                        count_bigram[front_word] = defaultdict(float)
                        count_bigram[front_word][cur_word] += 1
                    index += 1

            write_to_json(count_bigram, self.bigram_file)
            return count_bigram
        else:
            print("bi-gram模型已存在，正在加载...")
            return self.__load_model()

    def __load_training_data(self):

        all_tokens = []
        with open(self.corpus, 'r', encoding='utf8') as f:
            for line in f:
                tokens = re.split(r'\s+', line.strip())
                all_tokens.append(tokens)

        return all_tokens

    def __load_model(self):
        '''
        加载概率模型
        :param model_file:
        :return:
        '''
        count_bigram = read_from_json(self.bigram_file)
        return count_bigram

    def __load_word_dict(self):
        '''
        读取词典
        :param word_dict:
        :return:
        '''
        word_dict = read_from_json(self.word_dict_file)
        return word_dict

    def __wordtypes(self):
        '''
        获取word types
        :return:
        '''
        return len(self.word_dict)

    def __cal_sentence_prob(self, splits):
        '''
        使用加一平滑方法，计算得出最大概率分词序列
        :param splits: 分法组合
        :return: 最佳分词方案
        '''
        all_p = []
        already_p = {}
        max_p = -99999
        for split in splits:
            # 计算概率 为了防止溢出 对概率取对数，转换乘法为加法
            index = 0
            P_of_segment = 0
            while index < len(split) - 1:
                cur_word = split[index + 1]
                front_word = split[index]
                key = cur_word + '|' + front_word

                if key in already_p:
                    P_of_segment += already_p[key]
                    index += 1
                    continue

                if front_word in self.bigram and cur_word in self.bigram[front_word]:
                    p = math.log(
                        ((self.bigram[front_word][cur_word] + 1) / (self.word_dict[front_word] + self.__wordtypes())))

                    P_of_segment += p

                else:
                    if front_word not in self.word_dict:
                        p = math.log(1 / self.__wordtypes())
                    else:
                        p = math.log((1 / (self.word_dict[front_word] + self.__wordtypes())))

                    P_of_segment += p

                already_p[key] = p
                index += 1
            all_p.append(P_of_segment)
        max_index = 0
        for idx, P in enumerate(all_p):
            if max_p < P:
                max_p = P
                max_index = idx

        return splits[max_index]

    def __segment_sentence(self, sentence):
        '''
        对一句话进行分词
        :param sentence:
        :return:
        '''

        splits = self.__construct_segment(sentence=sentence)

        best = self.__cal_sentence_prob(splits=splits)

        return best

    def segment(self, file):
        '''
        n-gram模型分词接口
        :param file:
        :return:
        '''

        segment_words = []
        with open(file, 'r', encoding='utf8') as f:
            context = f.readlines()
            for i in tqdm(range(len(context))):

                segment_words.append(self.__segment_sentence(context[i].strip()))

        return segment_words


class HMM(object):
    STATUS = (_B, _E, _M, _S) = range(4)
    LOG_NEGATIVE_INF = -3.14e+100

    def __init__(self, corpus=TRAINING_FILE):

        # 发射矩阵
        '''
        [
         [0:B]{'wb1':pb1, 'wb2':pb2, ...}
         [1:E]{'we1':pe1, 'we2':pe2, ...}
         [2:M]{'wm1':pm1, 'wm2':pm2, ...}
         [3:S]{'ws1':ps1, 'ws2':ps2, ...}
        ]
        '''
        self.emission_matrix = [{}, {}, {}, {}]
        # 状态转移概率矩阵
        '''
          B  E  M  S
      B  [[0, 0, 0, 0], 
      E  [0, 0, 0, 0],
      M  [0, 0, 0, 0],
      S  [0, 0, 0, 0]]
        '''
        self.trans_matrix = np.zeros(shape=(4, 4))

        # 初始状态概率向量
        '''
         B  E  M  S
        [0, 0, 0, 0]
        '''
        self.init_status_prob = np.zeros(shape=(4,))
        self.init_model_data(file=corpus)

    def __str__(self):
        return 'Hidden Markov Model'

    def init_model_data(self, file):

        if os.path.exists(TRANS_MATRIX) and os.path.exists(INIT_STATUS) and os.path.exists(EMISSION_MATRIX):

            print("hmm模型已存在，正在加载...")
            self.__load_model_data()
        else:
            print("正在训练...")
            self.__training(file=file)

    def __training(self, file):
        '''
        训练HMM模型
        :param file:
        :return:
        '''
        with open(file, 'r', encoding='utf8') as f:
            for line in f:
                if len(line.strip()) == 0:
                    continue
                sentence_status = []
                words = list(filter(lambda word: len(word) > 0, line.strip().split()))
                for word in words:

                    if len(word) == 1:
                        sentence_status.append(self._S)
                    else:
                        sentence_status.append(self._B)
                        if len(word) == 2:
                            sentence_status.append(self._E)
                        else:
                            times = len(word) - 2
                            while times > 0:
                                sentence_status.append(self._M)
                                times -= 1
                            sentence_status.append(self._E)
                self.init_status_prob[sentence_status[0]] += 1
                pairs = list(zip("".join(words), sentence_status))

                for i in range(len(pairs)):
                    cur_pair = pairs[i]
                    last_pair = pairs[i - 1]
                    self.trans_matrix[last_pair[1]][cur_pair[1]] += 1
                    self.__change_dict_value(self.emission_matrix[cur_pair[1]], cur_pair[0])

        self.__cal_all_prob()
        self.__save_model_data()

    def __cal_all_prob(self):
        '''
        计算全部的概率
        :return:
        '''
        # 计算发射矩阵
        for emission in self.emission_matrix:
            total_cnt = sum(emission.values())
            for key in emission.keys():
                emission[key] = self.LOG_NEGATIVE_INF if emission[key] == 0 \
                    else math.log(emission[key] / total_cnt)

        # 计算转移概率的log值
        for trans in self.trans_matrix:
            total = sum(trans)
            for i in range(len(trans)):
                trans[i] = self.LOG_NEGATIVE_INF if trans[i] == 0 \
                    else math.log(trans[i] / total)

        # 计算初始状态概率的log值
        total = sum(self.init_status_prob)
        for i in range(len(self.init_status_prob)):
            self.init_status_prob[i] = self.LOG_NEGATIVE_INF if self.init_status_prob[i] == 0 \
                else math.log(self.init_status_prob[i] / total)

    def __viterbi_cut(self, sentence):
        '''
        viterbi算法
        :return:
        '''

        # 某状态下，观察值的概率
        weight = np.zeros(shape=(4, len(sentence)))

        # 权重取最大时，前一个观察值的状态
        path = np.zeros(shape=(4, len(sentence)), dtype=np.int)

        # 初始化第一个字
        first_word = sentence[0]

        for s in self.STATUS:
            # 计算条件概率， 给定某状态下，观察值为第一字的概率，这里做加法是因为所有的概率全部都取得log值
            weight[s][0] = self.init_status_prob[s] + self.emission_matrix[s].get(first_word, self.LOG_NEGATIVE_INF)

        for word_idx in range(1, len(sentence)):
            for s in self.STATUS:
                weight[s][word_idx] = self.LOG_NEGATIVE_INF
                path[s][word_idx] = -1
                for i in range(4):
                    last = weight[i][word_idx - 1] + self.trans_matrix[i][s] + \
                           self.emission_matrix[s].get(sentence[word_idx], self.LOG_NEGATIVE_INF)
                    if last > weight[s][word_idx]:
                        weight[s][word_idx] = last
                        path[s][word_idx] = i

        back_through = int(np.argmax(weight, axis=0)[len(sentence) - 1])
        status_result = []
        for idx in range(len(sentence) - 1, -1, -1):
            status_result.append(back_through)
            back_through = path[back_through][idx]
        status_result.reverse()
        return self.__insert_seg_char(sentence, status_result)

    def __insert_seg_char(self, text, status, seg_char=' '):
        result = ""
        for (c, s) in zip(text, status):
            result += c
            if s in (self._E, self._S):
                result += seg_char
        return result

    def __change_dict_value(self, d, key):
        '''
        如果概率表中有这个词，计数加1，如果没有，设置为1
        :param d:
        :param key:
        :return:
        '''
        v = d.setdefault(key, 0)
        d[key] = v + 1

    def __save_model_data(self):
        # 保存发射矩阵
        write_to_json(json_data=self.emission_matrix, data_file=EMISSION_MATRIX)
        # 保存状态转移矩阵
        np.savetxt(TRANS_MATRIX, self.trans_matrix)
        np.savetxt(INIT_STATUS, self.init_status_prob)

    def __load_model_data(self):
        # 保存发射矩阵
        self.emission_matrix = read_from_json(data_file=EMISSION_MATRIX)
        # 保存状态转移矩阵
        self.trans_matrix = np.loadtxt(TRANS_MATRIX)
        self.init_status_prob = np.loadtxt(INIT_STATUS)

    def segment(self, file):
        '''
        HMM分词接口
        :param file:
        :return:
        '''
        segment_words = []
        with open(file, 'r', encoding='utf8') as f:
            for line in f:
                segment_words.append(re.split(r'\s+', self.__viterbi_cut(line.strip()).strip()))

        return segment_words


class MechanicalSegmentation(object):
    '''
    机械分词算法，正向最大，反向最大
    '''
    __WORD_DIC = read_from_json(WORD_DICT_FILE)

    def __init__(self, is_backward=True, max_len_of_word=MAX_LEN_OF_WORD):
        self.is_backward = is_backward
        self.max_len_of_word = max_len_of_word

    def __str__(self):
        return 'backward_maximum_match' if self.is_backward else 'forward_maximum_match'

    def __forward_maximum_match(self, sentence):
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

            if tail_index >= self.max_len_of_word:
                tail_index = self.max_len_of_word

            look_like_word = origin_sentence[index:tail_index]
            while tail_index > 1:

                if look_like_word not in self.__WORD_DIC:
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

        return segment

    def __backward_maximum_match(self, sentence):
        '''
        反向最大匹配
        :param sentence:输入句子
        :return:
        '''
        origin_sentence = sentence
        segment = []

        while origin_sentence != '':
            head_index = 0
            index = len(origin_sentence)
            if (index - head_index) >= self.max_len_of_word:
                head_index = index - self.max_len_of_word

            look_like_word = origin_sentence[head_index:index]
            while (index - head_index) > 1:

                if look_like_word not in self.__WORD_DIC:
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
                origin_sentence = origin_sentence[0:head_index - 1]
        segment = segment[::-1]
        return segment

    def segment(self, file):
        '''
        机械分词算法分词接口,对一段文本分词
        :param file: 要分词的语料
        :return:
        '''
        segment_words = []
        with open(file, 'r', encoding='utf8') as f:
            if self.is_backward:
                for line in f:
                    segment_words.append(self.__backward_maximum_match(line.strip()))
            else:
                for line in f:
                    segment_words.append(self.__forward_maximum_match(line.strip()))

        return segment_words


