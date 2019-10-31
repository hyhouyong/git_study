#!/usr/bin/python
# -*- coding: UTF-8 -*-
import cv2
import os
import sys


def clear_line(ic_path):
    img1 = cv2.imread(ic_path)[37:437, 0:1920]
    return img1


def Ato_run(loadpath, savepath):
    load_path = loadpath
    save_path = savepath
    for i in os.listdir(load_path):
        iw = i.split('.')[0]
        ni = '{}.jpg'.format(iw)
        picload_path = os.path.join(load_path, i)
        picsave_path = os.path.join(save_path, ni)
        imge2 = clear_line(picload_path)
        cv2.imwrite(picsave_path, imge2)


if __name__ == '__main__':
    load_path1 = './1'
    save_path1 = './2'
    Ato_run(load_path1, save_path1)
