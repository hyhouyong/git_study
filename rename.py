#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
此脚本需要在python3下执行,将中文名字转换成拼音
"""
from xpinyin import Pinyin
import os
PREFIX_NAME = ''

def list_all(dir_name):
    pinyin_converter = Pinyin()
    for dirpath, dirnames, filenames in os.walk(dir_name):
        for filename in filenames:
            rest = pinyin_converter.get_pinyin(filename, '')
            length = len(rest)
            res = rest[0:length-4]
            #res1 = res.split(".")[-2:0]
            res2 = rest.split(".")[-1]


            print(rest)
            print(res)
            #print(res1)
            print(res2)
            print(length)

            res = remove_chars(res,'-', '(', ')', '（', '）','、','：','，',',',' ')
            res = res.replace(".","_")
            print(res)
            if PREFIX_NAME != '':
                res = PREFIX_NAME + res
            if res[-5] == '_':
                res = res[:-5] + res[-4:]
            src_path = dirpath + '\\' + filename
            dest_path = dirpath + '\\' + res + "." + res2
            print(src_path + '->' + res)
            os.rename(src_path, dest_path)

def remove_chars(filename, *chars):
    for i in range(len(chars)):
        filename = filename.replace(chars[i], '')
    return filename
if __name__ == '__main__':
    dir_name = './1'  #需要改名的图片输入路径以及保存路径
    list_all(dir_name)






























