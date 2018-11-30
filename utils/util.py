# @File  : util.py
# @Author: Cuiqingyao
# @Date  : 18-11-29
# @Desc  : 
# @Contact: qingyaocui@gmail.com

import json
import os

def write_to_json(json_data, data_file):
    '''
    序列化json数据，易读格式
    :param json_data:
    :param data_file:
    :return:
    '''
    with open(data_file, 'w', encoding='utf8') as f:
        json.dump(obj=json_data, fp=f, ensure_ascii=False, indent=4, separators=(',', ":"))

def write_to_json_min(json_data, data_file):
    '''
    序列化json数据
    :param json_data:
    :param data_file:
    :return:
    '''
    with open(data_file, 'w', encoding='utf8') as f:
        json.dump(obj=json_data, fp=f, ensure_ascii=False)

def read_from_json(data_file):
    '''
    读取json数据
    :param data_file:
    :return:
    '''
    with open(data_file, 'r', encoding='utf8') as f:
        json_data = json.load(f)

    return json_data

def clean_data(corpus, clean_corpus):
    '''
    清洗数据，除去空行，并为每句话增加起始和终止标记
    :param corpus:
    :param clean_corpus:
    :return:
    '''
    if os.path.exists(corpus):
        if not os.path.exists(clean_corpus):
            with open(corpus, 'r', encoding='utf8') as f_in:
                with open(clean_corpus, 'w', encoding='utf8') as f_out:
                    for line in f_in:
                        if len(line.split()) != 0:
                            f_out.write('<B>/BEG  ' + line.strip() + '  <E>/END\n')
    else:
        raise Exception('原始数据路径不存在！')