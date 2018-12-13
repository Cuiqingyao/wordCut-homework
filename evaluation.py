# @File  : 
# @Author: Cuiqingyao
# @Date  : 2018/12/11
# @Desc  : 
# @Contact: qingyaocui@gmail.com

from cut_model import HMM, MechanicalSegmentation, Bigram
from  utils.util import read_from_txt
from settings import *

class statistic(object):
    """
    计算模型的准确率，召回率，F1值
    """
    def __init__(self, model):
        # 要统计的模型
        self.model = model()
        # 原始词数
        self.origin_words_num = 0
        # 总分词数
        self.total_seg_words_num = 0
        # 正确分词数
        self.correct_words_num = 0
        # 错误分词数
        self.incor_words_num = 0

    def __statistic(self, test_data_file, answer_file):
        '''
        基于测试数据，统计分词总数，原始词数，正确分词数，错误分词数
        :param test_data_file:
        :param answer_file:
        :return:
        '''

        answers = read_from_txt(answer_file)
        segment_words = self.model.segment(file=test_data_file)

        # 判断是否完成完整的分词，分词的答案行数与模型分词的行数必须一致
        # assert len(answers) == len(segment_words)


        # 统计原始分词数
        for answer in answers:
            self.origin_words_num += len(answer)

        # 统计分词总数
        for words in segment_words:
            self.total_seg_words_num += len(words)

        # 统计正确分词个数 和 错误分词个数
        for idx, answer in enumerate(answers):
            for word in segment_words[idx]:
                if word in answer:
                    self.correct_words_num += 1
                else:
                    self.incor_words_num += 1


        # 保证正确分词个数加错误分词个数 与 分词总数相等
        assert self.total_seg_words_num == self.correct_words_num + self.incor_words_num

    def precision(self):
        '''
        计算准确率
        :return:
        '''
        return self.correct_words_num / self.origin_words_num
    def recall(self):
        '''
        计算召回率
        :return:
        '''
        return self.correct_words_num / self.total_seg_words_num
    def F1(self):
        '''
        计算F1值
        :return:
        '''
        return 2 * self.precision() * self.recall() / (self.precision() + self.recall())

    def print_report(self):
        '''
        打印统计报告
        :return:
        '''
        print('当前分词模型:',self.model)
        print('原始分词数量(origin_words_num):', self.origin_words_num)
        print('总分词数量(total_seg_words_num):', self.total_seg_words_num)
        print('正确分词数量(correct_words_num):', self.correct_words_num)
        print('错误分词数量(incor_words_num):', self.incor_words_num)
        print('准确率(precision)：', self.precision())
        print('召回率(recall)：', self.recall())
        print('F1值：', self.F1())

    def evaluation(self, test_data_file, answer_file):
        '''
        评估程序接口
        :param test_data_file: 测试数据
        :param answer_file: 正确分词答案
        :return:
        '''
        self.__statistic(test_data_file=test_data_file, answer_file=answer_file)
        self.print_report()

if __name__ == '__main__':
    stt = statistic(model=Bigram)
    stt.evaluation(test_data_file=TEST_FILE, answer_file=ANSWERS)