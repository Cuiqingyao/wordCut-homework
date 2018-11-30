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

