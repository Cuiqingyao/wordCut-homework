# @File  : cut.py
# @Author: Cuiqingyao
# @Date  : 18-11-29
# @Desc  : 
# @Contact: qingyaocui@gmail.com

from cut_model import MechanicalSegmentation
from settings import *

def main(M, target=TARGET_FILE, result_file=RESULT_FILE):
    '''
    分词作业入口程序
    :param M: 使用的分词模型
    :param target: 测试数据集
    :param result_file: 分词结构文件 2018140546.txt
    :return:
    '''
    model = M()
    results = model.segment(file=target)
    with open(result_file, 'w', encoding='utf8') as f:
        for result in results:
            f.write(' '.join(result) + '\n')


if __name__ == '__main__':

    main(M=MechanicalSegmentation)
