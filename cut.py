# @File  : cut.py
# @Author: Cuiqingyao
# @Date  : 18-11-29
# @Desc  : 
# @Contact: qingyaocui@gmail.com

from cut_model import Ngram, HMM, MechanicalSegmentation
from settings import RESULT_FILE, TARGET_FILE

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
            f.write(' '.join(result[1:-1]) + '\n')


if __name__ == '__main__':
    main(M=Ngram)
    # main(M=HMM)
    # main(M=MechanicalSegmentation)
