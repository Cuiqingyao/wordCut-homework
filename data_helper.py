# @File  : data_helper.py
# @Author: Cuiqingyao
# @Date  : 18-11-29
# @Desc  : 
# @Contact: qingyaocui@gmail.com


import math
from util import *
from settings import *





class DataHelper(object):
    '''
    数据预处理
    '''

    def __init__(self, origin_corpus=ORIGIN_CORPUS, word_dict_file=WORD_DICT_FILE):

        # 原始语料
        self.origin_corpus = origin_corpus
        # 字典
        self.word_dict_file = word_dict_file

    def __str__(self):

        return 'datahelper -- origin_corpus :{0}' % (self.origin_corpus)

    def __build_dict(self):
        '''
        构建字典
        :return:
        '''

        try:

            if os.path.exists(TRAINING_FILE):
                word_dict = {}
                training_data = self.__load(type='txt', data_file=TRAINING_FILE)
                for data in training_data:
                    for word in data:
                        self.__update_count(d=word_dict, key=word)
                self.__save(data=word_dict, type='json', data_file=self.word_dict_file)

                # 并统计最大词长，word_type数量, token数据
                # 最大词长
                self.__max_word_length(word_dict=word_dict)
                # word types
                self.__number_of_word_types(word_dict=word_dict)
                # tokens
                self.__number_of_tokens(training_data=training_data)

            # 基于全语料库的构建词典
            if os.path.exists(TRAINING_FILE_FOR_BIGRAM):
                word_dict_bigram = {}
                bi_gram_data = self.__load(type='txt', data_file=TRAINING_FILE_FOR_BIGRAM)
                for data in bi_gram_data:
                    for word in data:
                        self.__update_count(d=word_dict_bigram, key=word)
                self.__save(data=word_dict_bigram, type='json', data_file=WORD_DICT_FILE_FOR_BIGRAM)
        except Exception as e:
            print(e)
            print("程序正在推出...")
            exit()

    def __save(self, data, type, data_file):
        '''
        保存数据
        :param data:
        :param type: 数据格式
        :param data_file:
        :return:
        '''
        if type == 'json':
            write_to_json(json_data=data, data_file=data_file)
        elif type == 'txt':
            write_to_txt(txt_data=data, data_file=data_file)
        else:
            raise Exception("不支持的格式%s!" % (type))

    def __load(self, type, data_file):
        '''
        加载数据
        :param type: 数据格式
        :param data_file:
        :return:
        '''
        if type == 'json':
            return read_from_json(data_file=data_file)
        elif type == 'txt':
            return read_from_txt(data_file=data_file)
        else:
            raise Exception("不支持的格式%s!" % (type))

    def __clean_corpus(self):
        '''
        语料清洗，去除空行，去除每一行的日期
        :return: 干净语料
        '''
        clean_data = []
        if os.path.exists(self.origin_corpus):
            with open(self.origin_corpus, 'r', encoding='utf8') as f_in:
                for line in f_in:
                    # 对于不是空行，添加句子头标记和尾标记，并去除词性，和日期
                    if len(line.split()) != 0:
                        data = []
                        pairs = re.split(r'\s+', line.strip())
                        for pair in pairs[1:]:
                            token, _ = pair.split('/')
                            data.append(token)
                        clean_data.append(data)
        else:
            raise Exception('原始语料不存在！')

        return clean_data

    def __update_count(self, d, key):
        '''
        更新字典值，每次加1
        :param d:
        :param key:
        :return:
        '''
        v = d.setdefault(key, 0)
        d[key] = v + 1

    def prcessed(self):
        '''
        数据处理接口
        :return:
        '''
        if os.path.exists(TRAINING_FILE) and \
            os.path.exists(TEST_FILE) and \
            os.path.exists(WORD_DICT_FILE) and\
            os.path.exists(TRAINING_FILE_FOR_BIGRAM):
            print('数据已经处理完毕...')
        else:
            print('数据预处理  ... ')
            print('生成训练数据和测试数据  ...')
            print('training data size : test data size = %.1f : %.1f' % (TRAINING, TEST))

            # 生成训练数据 测试数据 分词答案
            self.__generate_train_test_data()

            # 针对训练数据生成字典
            self.__build_dict()

            print("数据生成完成  ...")

    def __max_word_length(self, word_dict):
        '''
        计算最大词长
        :return:
        '''
        max_len = 0
        for word in word_dict:
            if max_len < len(word):
                max_len = len(word)
        print('max word length :', max_len)

    def __number_of_word_types(self, word_dict):
        '''
        计算word type的数量
        :return:
        '''
        print('number of word types :', len(word_dict))

    def __number_of_tokens(self, training_data):
        '''
        计算token的数量
        :param training_data:
        :return:
        '''
        tokens_num = 0
        for data in training_data:
            tokens_num += len(data)
        print('number of tokens :', tokens_num)

    def __generate_train_test_data(self):
        '''
        生成训练数据和测试数据
        :return:
        '''
        assert TRAINING < 1.0 and TEST < 1.0 and (TRAINING + TEST) == 1.0
        try:
            clean_data = self.__clean_corpus()

            training_size = math.ceil(TRAINING * len(clean_data))
            training_data = clean_data[0:training_size]
            answers = clean_data[training_size:]
            # 保存训练数据
            self.__save(data=training_data, type='txt', data_file=TRAINING_FILE)
            # 保存分词答案
            self.__save(data=answers, type='txt', data_file=ANSWERS)
            test_data = []
            for answer in answers:
                sentence = ''
                for word in answer:
                    sentence += word
                test_data.append(sentence)
            write_sentences(txt_data=test_data, data_file=TEST_FILE)

            # 生成整个语料数据用于训练bigram
            for data in clean_data:
                data.insert(0, '<B>')
            self.__save(data=clean_data, type='txt', data_file=TRAINING_FILE_FOR_BIGRAM)

        except Exception as e:
            print(e)
            print("程序正在推出...")
            exit()

if __name__ == '__main__':
    datahelper = DataHelper()
    datahelper.prcessed()


