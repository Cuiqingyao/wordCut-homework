# @File  : util.py
# @Author: Cuiqingyao
# @Date  : 18-11-29
# @Desc  : 
# @Contact: qingyaocui@gmail.com

import json
import re

def write_to_json(json_data, data_file):
    '''
    序列化json数据，易读格式
    :param json_data:
    :param data_file:
    :return:
    '''
    with open(data_file, 'w', encoding='utf8') as f:
        json.dump(obj=json_data, fp=f, ensure_ascii=False)

def write_to_txt(txt_data, data_file):
    '''
    将数据写入txt文件,分行
    :param txt_data:
    :param data_file:
    :return:
    '''
    with open(data_file, 'w', encoding='utf8') as f:
        for data in txt_data:
            f.write(' '.join(data) + '\n')

def write_sentences(txt_data, data_file):
    '''
    将数据写入txt文件,分行
    :param txt_data:
    :param data_file:
    :return:
    '''
    with open(data_file, 'w', encoding='utf8') as f:
        for data in txt_data:
            f.write(data + '\n')

def writer_answer_to_txt(results, file):
    '''
    分词结果写入文件
    :param data:
    :param file:
    :return:
    '''

    with open(file, 'w', encoding='utf8') as f:
        if isinstance(results[0], list):
            for result in results:
                f.write(' '.join(result[1:-1]) + '\n')
        elif isinstance(results[0], str):
            for result in results:
                f.write(result + '\n')
        else:
            raise Exception("结果格式错误！")


def read_from_json(data_file):
    '''
    读取json数据
    :param data_file:
    :return:
    '''
    with open(data_file, 'r', encoding='utf8') as f:
        json_data = json.load(f)

    return json_data

def read_from_txt(data_file):
    '''
    读取语料数据
    :param data_file: 语料
    :return: list, 每个元素对用一行分词数据
    '''
    data = []
    with open(data_file, 'r', encoding='utf8') as f:
       for line in f:
           data.append(re.split(r'\s+',line.strip()))
    return data
