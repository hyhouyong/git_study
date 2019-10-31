# -- coding:UTF-8 --
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import pylab
import datetime
from PIL import Image
import os
import hy_zj
import cv2
import add_top
import time
# the path of caffe & the path of python in caffe
import sys, getopt
import os
from multiprocessing import Process
from multiprocessing import Manager
sys.path.append('/home/hjin/caffe-ssd/python')
import caffe

caffe_root = '/home/nvidia/caffe-ssd/'

# caffe.set_mode_cpu()
#caffe.set_mode_gpu()
#caffe.set_device(2)

from google.protobuf import text_format
from caffe.proto import caffe_pb2

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

model_def = 'models/deploy.prototxt'
model_weights = 'models/VGG_9_2_11label_wave_SSD_800x400_iter_30000.caffemodel'
labelmap_file = 'models/labelmap_voc.prototxt'


# get parameter
def GetJPGName(InputImagePath):
    jpg_names = []
    jpgs = os.listdir(InputImagePath)
    for one_jpg in jpgs:
        if (os.path.splitext(one_jpg)[1] == '.jpg'):
            jpg_names.append(one_jpg)
    return jpg_names


def WrongInput():
    print("Usage: python slope_detection.py -i InputImagePath -o OutputImagePath -s score -l OutputCSVFileName")
    sys.exit(1)


def GetPara():
    try:
        opts, args = getopt.getopt(sys.argv[1:], "i:o:s:l:")
    except getopt.GetoptError:
        WrongInput()
    # get options into dictionary
    option = []
    value = []
    for opt, val in opts:
        option.append(opt)
        value.append(val)
    parameter = dict(zip(option, value))

    # check para
    if len(parameter.keys()) != 4:
        WrongInput()
    return parameter['-i'], parameter['-o'], parameter['-s'], parameter['-l']


# about result.csv
def InitCSV(OutputCSVFileName):
    WriteCSVFile = open(OutputCSVFileName, 'w')
    WriteCSVFile.write(",".join(['0', '1', '2', '3', '4', '5', '6']) + "\n")
    WriteCSVFile.write(",".join(['image_name', 'x1', 'y1', 'x2', 'y2', 'label', 'score']) + "\n")
    WriteCSVFile.close()


# def WriteinCSV(OutputCSVFileName, image_name, xmin, ymin, xmax, ymax, label_name, score):
#     WriteCSVFile = open(OutputCSVFileName, 'a')
#     one_line_data = [image_name, xmin, ymin, xmax, ymax, label_name, score]
#     WriteCSVFile.write(",".join(one_line_data) + "\n")
#     WriteCSVFile.close()

def WriteinCSV(OutputCSVFileName, L):
    WriteCSVFile = open(OutputCSVFileName, 'a')
    for pi in L:
        for p1_image_i in pi:
            for p1_imagei_Labels in p1_image_i:
                for p1_imagei_Label in p1_imagei_Labels:
                    WriteCSVFile.write(",".join(p1_imagei_Label) + "\n")
    WriteCSVFile.close()

def RunDetection1(InputImagePath, OutputImagePath, jpg_names, score_threshold,return_list):
    p1t1 = time.time()
    caffe.set_mode_gpu()
    caffe.set_device(0)
    net = caffe.Net(model_def, model_weights, caffe.TEST)
    # print('jpg_names:{}'.format(jpg_names))
    p1 = []
    for one_jpg in jpg_names:
        p1_image_i = []

        name = one_jpg.split('.jpg')[0]
        load_path = os.path.join(InputImagePath + '/' + one_jpg)
        load_path1 = os.path.join(InputImagePath + '/' + name + '_1.jpg')
        load_path2 = os.path.join(InputImagePath + '/' + name + '_2.jpg')
        load_path3 = os.path.join(InputImagePath + '/' + name + '_3.jpg')
        img = Image.open(load_path)
        sp = img.size
        x = sp[0] // 2 - 400
        y = x + 800
        img1 = img.crop((0, 0, 800, sp[1]))
        img1.save(load_path1)
        save_path = os.path.join(OutputImagePath + '/' + one_jpg)
        pic_name = one_jpg.split('.')[0]
        save_pic_name = '{}.bmp'.format(pic_name)

        image = caffe.io.load_image(load_path1)
        transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
        transformer.set_transpose('data', (2, 0, 1))
        transformer.set_mean('data', np.array([104, 117, 123]))  # mean pixel
        transformer.set_raw_scale('data',
                                  255)  # the reference model operates on images in [0,255] range instead of [0,1]
        transformer.set_channel_swap('data', (2, 1, 0))  # the reference model has channels in BGR order instead of RGB
        # image_resize = 512
        # net.blobs['data'].reshape(1, 3, image_resize, image_resize)
        net.blobs['data'].reshape(1, 3, 400, 800)
        transformed_image = transformer.preprocess('data', image)

        # transformed_image = detection_init(labelmap_file, net, image)
        net.blobs['data'].data[...] = transformed_image
        detections = net.forward()['detection_out']

        # Parse the outputs.
        det_label = detections[0, 0, :, 1]
        det_conf = detections[0, 0, :, 2]
        det_xmin = detections[0, 0, :, 3]
        det_ymin = detections[0, 0, :, 4]
        det_xmax = detections[0, 0, :, 5]
        det_ymax = detections[0, 0, :, 6]

        # Get detections with confidence higher than 0.6.
        top_indices = [i for i, conf in enumerate(det_conf) if conf >= score_threshold]
        top_conf = det_conf[top_indices]
        top_label_indices = det_label[top_indices].tolist()

        # get_lablename_fcn
        file = open(labelmap_file, 'r')
        labelmap = caffe_pb2.LabelMap()
        text_format.Merge(str(file.read()), labelmap)
        num_labels = len(labelmap.item)
        labelnames = []


        for label in top_label_indices:
            found = False
            for i in range(0, num_labels):
                if label == labelmap.item[i].label:
                    found = True
                    labelnames.append(labelmap.item[i].display_name)
                    break
            assert found == True

        top_labels = labelnames
        top_xmin = det_xmin[top_indices]
        top_ymin = det_ymin[top_indices]
        top_xmax = det_xmax[top_indices]
        top_ymax = det_ymax[top_indices]
        # Plot the boxes in test picture

        colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
        plt.switch_backend('agg')
        currentAxis = plt.gca()
        box_label = []
        box_xmin = []
        box_ymin = []
        box_xmax = []
        box_ymax = []
        box_score = []

        p1_image1_Labels = []
        for i in range(top_conf.shape[0]):
            # 截图图片的一个标签
            p1_image1_Label = []
            # bbox value
            xmin = int(round(top_xmin[i] * image.shape[1]))
            ymin = int(round(top_ymin[i] * image.shape[0]))
            xmax = int(round(top_xmax[i] * image.shape[1]))
            ymax = int(round(top_ymax[i] * image.shape[0]))
            if xmin < 0:
                xmin = 0
            if ymin < 0:
                ymin = 0
            if xmax > image.shape[1]:
                xmax = image.shape[1]
            if ymax > image.shape[0]:
                ymax = image.shape[0]
            # score
            score = top_conf[i]
            # label
            label = int(top_label_indices[i])
            label_name = top_labels[i]
            box_label.append(str(label_name))
            box_xmin.append(xmin)
            box_ymin.append(ymin)
            box_xmax.append(xmax)
            box_ymax.append(ymax)
            box_score.append(score)

            p1_image1_Label.append(save_pic_name)
            p1_image1_Label.append(str(xmin))
            p1_image1_Label.append(str(ymin))
            p1_image1_Label.append(str(xmax))
            p1_image1_Label.append(str(ymax))
            p1_image1_Label.append(label_name)
            p1_image1_Label.append(str(score))

            p1_image1_Labels.append(p1_image1_Label)

            # WriteinCSV("1.csv", save_pic_name, str(xmin), str(ymin), str(xmax), str(ymax),label_name,str(score))
        p1_image_i.append(p1_image1_Labels)

        img2 = img.crop((x, 0, y, sp[1]))
        img2.save(load_path2)
        image = caffe.io.load_image(load_path2)

        # net = caffe.Net(model_def, model_weights, caffe.TEST)
        transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
        transformer.set_transpose('data', (2, 0, 1))
        transformer.set_mean('data', np.array([104, 117, 123]))  # mean pixel
        transformer.set_raw_scale('data',
                                  255)  # the reference model operates on images in [0,255] range instead of [0,1]
        transformer.set_channel_swap('data',
                                     (2, 1, 0))  # the reference model has channels in BGR order instead of RGB
        # image_resize = 512
        # net.blobs['data'].reshape(1, 3, image_resize, image_resize)
        net.blobs['data'].reshape(1, 3, 400, 800)
        transformed_image = transformer.preprocess('data', image)

        # transformed_image = detection_init(labelmap_file, net, image)
        net.blobs['data'].data[...] = transformed_image
        detections = net.forward()['detection_out']

        # Parse the outputs.
        det_label = detections[0, 0, :, 1]
        det_conf = detections[0, 0, :, 2]
        det_xmin = detections[0, 0, :, 3]
        det_ymin = detections[0, 0, :, 4]
        det_xmax = detections[0, 0, :, 5]
        det_ymax = detections[0, 0, :, 6]

        # Get detections with confidence higher than 0.6.
        top_indices = [i for i, conf in enumerate(det_conf) if conf >= score_threshold]
        top_conf = det_conf[top_indices]
        top_label_indices = det_label[top_indices].tolist()

        # get_lablename_fcn
        file = open(labelmap_file, 'r')
        labelmap = caffe_pb2.LabelMap()
        text_format.Merge(str(file.read()), labelmap)
        num_labels = len(labelmap.item)
        labelnames = []

        for label in top_label_indices:
            found = False
            for i in range(0, num_labels):
                if label == labelmap.item[i].label:
                    found = True
                    labelnames.append(labelmap.item[i].display_name)
                    break
            assert found == True

        top_labels = labelnames
        top_xmin = det_xmin[top_indices]
        top_ymin = det_ymin[top_indices]
        top_xmax = det_xmax[top_indices]
        top_ymax = det_ymax[top_indices]
        # Plot the boxes in test picture

        colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
        plt.switch_backend('agg')
        currentAxis = plt.gca()

        p1_image2_Labels = []
        for i in range(top_conf.shape[0]):

            p1_image2_Label = []

            # bbox value
            xmin = int(round(top_xmin[i] * image.shape[1]))
            ymin = int(round(top_ymin[i] * image.shape[0]))
            xmax = int(round(top_xmax[i] * image.shape[1]))
            ymax = int(round(top_ymax[i] * image.shape[0]))
            # score
            score = top_conf[i]
            # label
            label = int(top_label_indices[i])
            if xmin < 0:
                xmin = 0
            if ymin < 0:
                ymin = 0
            if xmax > image.shape[1]:
                xmax = image.shape[1]
            if ymax > image.shape[0]:
                ymax = image.shape[0]
            label_name = top_labels[i]
            box_label.append(str(label_name))
            box2_x1 = xmin + x
            box2_x2 = xmax + x
            box_xmin.append(box2_x1)
            box_ymin.append(ymin)
            box_xmax.append(box2_x2)
            box_ymax.append(ymax)
            box_score.append(score)

            p1_image2_Label.append(save_pic_name)
            p1_image2_Label.append(str(box2_x1))
            p1_image2_Label.append(str(ymin))
            p1_image2_Label.append(str(box2_x2))
            p1_image2_Label.append(str(ymax))
            p1_image2_Label.append(label_name)
            p1_image2_Label.append(str(score))

            p1_image2_Labels.append(p1_image2_Label)



            # WriteinCSV("1.csv", save_pic_name, str(box2_x1), str(ymin), str(box2_x2), str(ymax), label_name,str(score))
        p1_image_i.append(p1_image2_Labels)

        img3 = img.crop((sp[0] - 800, 0, sp[0], sp[1]))
        img3.save(load_path3)
        image = caffe.io.load_image(load_path3)

        # net = caffe.Net(model_def, model_weights, caffe.TEST)
        transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
        transformer.set_transpose('data', (2, 0, 1))
        transformer.set_mean('data', np.array([104, 117, 123]))  # mean pixel
        transformer.set_raw_scale('data',
                                  255)  # the reference model operates on images in [0,255] range instead of [0,1]
        transformer.set_channel_swap('data',
                                     (2, 1, 0))  # the reference model has channels in BGR order instead of RGB
        # image_resize = 512
        # net.blobs['data'].reshape(1, 3, image_resize, image_resize)
        net.blobs['data'].reshape(1, 3, 400, 800)
        transformed_image = transformer.preprocess('data', image)

        # transformed_image = detection_init(labelmap_file, net, image)
        net.blobs['data'].data[...] = transformed_image
        detections = net.forward()['detection_out']

        # Parse the outputs.
        det_label = detections[0, 0, :, 1]
        det_conf = detections[0, 0, :, 2]
        det_xmin = detections[0, 0, :, 3]
        det_ymin = detections[0, 0, :, 4]
        det_xmax = detections[0, 0, :, 5]
        det_ymax = detections[0, 0, :, 6]

        # Get detections with confidence higher than 0.6.
        top_indices = [i for i, conf in enumerate(det_conf) if conf >= score_threshold]
        top_conf = det_conf[top_indices]
        top_label_indices = det_label[top_indices].tolist()

        # get_lablename_fcn
        file = open(labelmap_file, 'r')
        labelmap = caffe_pb2.LabelMap()
        text_format.Merge(str(file.read()), labelmap)
        num_labels = len(labelmap.item)
        labelnames = []

        for label in top_label_indices:
            found = False
            for i in range(0, num_labels):
                if label == labelmap.item[i].label:
                    found = True
                    labelnames.append(labelmap.item[i].display_name)
                    break
            assert found == True

        top_labels = labelnames
        top_xmin = det_xmin[top_indices]
        top_ymin = det_ymin[top_indices]
        top_xmax = det_xmax[top_indices]
        top_ymax = det_ymax[top_indices]
        # Plot the boxes in test picture

        colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
        plt.switch_backend('agg')
        currentAxis = plt.gca()
        p1_image3_Labels = []
        for i in range(top_conf.shape[0]):
            # bbox value

            p1_image3_Label = []

            xmin = int(round(top_xmin[i] * image.shape[1]))
            ymin = int(round(top_ymin[i] * image.shape[0]))
            xmax = int(round(top_xmax[i] * image.shape[1]))
            ymax = int(round(top_ymax[i] * image.shape[0]))
            if xmin < 0:
                xmin = 0
            if ymin < 0:
                ymin = 0
            if xmax > image.shape[1]:
                xmax = image.shape[1]
            if ymax > image.shape[0]:
                ymax = image.shape[0]
            # score
            score = top_conf[i]
            # label
            label = int(top_label_indices[i])
            label_name = top_labels[i]
            box3_x1 = xmin + sp[0] - 800
            box3_x2 = xmax + sp[0] - 800
            box_label.append(str(label_name))
            box_xmin.append(box3_x1)
            box_ymin.append(ymin)
            box_xmax.append(box3_x2)
            box_ymax.append(ymax)
            box_score.append(score)


            p1_image3_Label.append(save_pic_name)
            p1_image3_Label.append(str(box3_x1))
            p1_image3_Label.append(str(ymin))
            p1_image3_Label.append(str(box3_x2))
            p1_image3_Label.append(str(ymax))
            p1_image3_Label.append(label_name)
            p1_image3_Label.append(str(score))


            p1_image3_Labels.append(p1_image3_Label)



            # WriteinCSV("1.csv", save_pic_name, str(box3_x1), str(ymin), str(box3_x2), str(ymax),label_name,str(score))
        p1_image_i.append(p1_image3_Labels)

        box = [box_label, box_xmin, box_ymin, box_xmax, box_ymax, box_score]

        try:
            box_dy, zj, box_s = add_top.main(load_path, box)
            if len(box_dy) > 0:
                p1_image_box = []
                for i in range(len(box_dy)):
                    Lbox = []
                    xmin = box_dy[i][0]
                    ymin = box_dy[i][1]
                    xmax = box_dy[i][2]
                    ymax = box_dy[i][3]
                    label_name = zj[i][0]
                    score = box_s[i][0]


                    Lbox.append(save_pic_name)
                    Lbox.append(str(xmin))
                    Lbox.append(str(ymin))
                    Lbox.append(str(xmax))
                    Lbox.append(str(ymax))
                    Lbox.append(label_name)
                    Lbox.append(str(score))

                    p1_image_box.append(Lbox)
                p1_image_i.append(p1_image_box)
                    # WriteinCSV("1.csv", save_pic_name, str(xmin), str(ymin), str(xmax), str(ymax), label_name,str(score))
            else:
                pass
        except TypeError:
            continue
        p1.append(p1_image_i)
    return_list.append(p1)
    p1t = time.time()-p1t1
    print("进程1时间：",p1t)
def RunDetection2(InputImagePath, OutputImagePath, jpg_names, score_threshold,return_list):
    p2t1 = time.time()
    caffe.set_mode_gpu()
    caffe.set_device(1)
    net = caffe.Net(model_def, model_weights, caffe.TEST)
    # print('jpg_names:{}'.format(jpg_names))
    p2 = []
    for one_jpg in jpg_names:
        p2_image_i = []

        name = one_jpg.split('.jpg')[0]
        load_path = os.path.join(InputImagePath + '/' + one_jpg)
        load_path1 = os.path.join(InputImagePath + '/' + name + '_1.jpg')
        load_path2 = os.path.join(InputImagePath + '/' + name + '_2.jpg')
        load_path3 = os.path.join(InputImagePath + '/' + name + '_3.jpg')
        img = Image.open(load_path)
        sp = img.size
        x = sp[0] // 2 - 400
        y = x + 800
        img1 = img.crop((0, 0, 800, sp[1]))
        img1.save(load_path1)
        save_path = os.path.join(OutputImagePath + '/' + one_jpg)
        pic_name = one_jpg.split('.')[0]
        save_pic_name = '{}.bmp'.format(pic_name)

        image = caffe.io.load_image(load_path1)
        transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
        transformer.set_transpose('data', (2, 0, 1))
        transformer.set_mean('data', np.array([104, 117, 123]))  # mean pixel
        transformer.set_raw_scale('data',
                                  255)  # the reference model operates on images in [0,255] range instead of [0,1]
        transformer.set_channel_swap('data', (2, 1, 0))  # the reference model has channels in BGR order instead of RGB
        # image_resize = 512
        # net.blobs['data'].reshape(1, 3, image_resize, image_resize)
        net.blobs['data'].reshape(1, 3, 400, 800)
        transformed_image = transformer.preprocess('data', image)

        # transformed_image = detection_init(labelmap_file, net, image)
        net.blobs['data'].data[...] = transformed_image
        detections = net.forward()['detection_out']

        # Parse the outputs.
        det_label = detections[0, 0, :, 1]
        det_conf = detections[0, 0, :, 2]
        det_xmin = detections[0, 0, :, 3]
        det_ymin = detections[0, 0, :, 4]
        det_xmax = detections[0, 0, :, 5]
        det_ymax = detections[0, 0, :, 6]

        # Get detections with confidence higher than 0.6.
        top_indices = [i for i, conf in enumerate(det_conf) if conf >= score_threshold]
        top_conf = det_conf[top_indices]
        top_label_indices = det_label[top_indices].tolist()

        # get_lablename_fcn
        file = open(labelmap_file, 'r')
        labelmap = caffe_pb2.LabelMap()
        text_format.Merge(str(file.read()), labelmap)
        num_labels = len(labelmap.item)
        labelnames = []

        for label in top_label_indices:
            found = False
            for i in range(0, num_labels):
                if label == labelmap.item[i].label:
                    found = True
                    labelnames.append(labelmap.item[i].display_name)
                    break
            assert found == True

        top_labels = labelnames
        top_xmin = det_xmin[top_indices]
        top_ymin = det_ymin[top_indices]
        top_xmax = det_xmax[top_indices]
        top_ymax = det_ymax[top_indices]
        # Plot the boxes in test picture

        colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
        plt.switch_backend('agg')
        currentAxis = plt.gca()
        box_label = []
        box_xmin = []
        box_ymin = []
        box_xmax = []
        box_ymax = []
        box_score = []

        p2_image1_Labels = []
        for i in range(top_conf.shape[0]):
            # 截图图片的一个标签
            p2_image1_Label = []
            # bbox value
            xmin = int(round(top_xmin[i] * image.shape[1]))
            ymin = int(round(top_ymin[i] * image.shape[0]))
            xmax = int(round(top_xmax[i] * image.shape[1]))
            ymax = int(round(top_ymax[i] * image.shape[0]))
            if xmin < 0:
                xmin = 0
            if ymin < 0:
                ymin = 0
            if xmax > image.shape[1]:
                xmax = image.shape[1]
            if ymax > image.shape[0]:
                ymax = image.shape[0]
            # score
            score = top_conf[i]
            # label
            label = int(top_label_indices[i])
            label_name = top_labels[i]
            box_label.append(str(label_name))
            box_xmin.append(xmin)
            box_ymin.append(ymin)
            box_xmax.append(xmax)
            box_ymax.append(ymax)
            box_score.append(score)

            p2_image1_Label.append(save_pic_name)
            p2_image1_Label.append(str(xmin))
            p2_image1_Label.append(str(ymin))
            p2_image1_Label.append(str(xmax))
            p2_image1_Label.append(str(ymax))
            p2_image1_Label.append(label_name)
            p2_image1_Label.append(str(score))

            p2_image1_Labels.append(p2_image1_Label)

            # WriteinCSV("1.csv", save_pic_name, str(xmin), str(ymin), str(xmax), str(ymax),label_name,str(score))
        p2_image_i.append(p2_image1_Labels)

        img2 = img.crop((x, 0, y, sp[1]))
        img2.save(load_path2)
        image = caffe.io.load_image(load_path2)

        # net = caffe.Net(model_def, model_weights, caffe.TEST)
        transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
        transformer.set_transpose('data', (2, 0, 1))
        transformer.set_mean('data', np.array([104, 117, 123]))  # mean pixel
        transformer.set_raw_scale('data',
                                  255)  # the reference model operates on images in [0,255] range instead of [0,1]
        transformer.set_channel_swap('data',
                                     (2, 1, 0))  # the reference model has channels in BGR order instead of RGB
        # image_resize = 512
        # net.blobs['data'].reshape(1, 3, image_resize, image_resize)
        net.blobs['data'].reshape(1, 3, 400, 800)
        transformed_image = transformer.preprocess('data', image)

        # transformed_image = detection_init(labelmap_file, net, image)
        net.blobs['data'].data[...] = transformed_image
        detections = net.forward()['detection_out']

        # Parse the outputs.
        det_label = detections[0, 0, :, 1]
        det_conf = detections[0, 0, :, 2]
        det_xmin = detections[0, 0, :, 3]
        det_ymin = detections[0, 0, :, 4]
        det_xmax = detections[0, 0, :, 5]
        det_ymax = detections[0, 0, :, 6]

        # Get detections with confidence higher than 0.6.
        top_indices = [i for i, conf in enumerate(det_conf) if conf >= score_threshold]
        top_conf = det_conf[top_indices]
        top_label_indices = det_label[top_indices].tolist()

        # get_lablename_fcn
        file = open(labelmap_file, 'r')
        labelmap = caffe_pb2.LabelMap()
        text_format.Merge(str(file.read()), labelmap)
        num_labels = len(labelmap.item)
        labelnames = []

        for label in top_label_indices:
            found = False
            for i in range(0, num_labels):
                if label == labelmap.item[i].label:
                    found = True
                    labelnames.append(labelmap.item[i].display_name)
                    break
            assert found == True

        top_labels = labelnames
        top_xmin = det_xmin[top_indices]
        top_ymin = det_ymin[top_indices]
        top_xmax = det_xmax[top_indices]
        top_ymax = det_ymax[top_indices]
        # Plot the boxes in test picture

        colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
        plt.switch_backend('agg')
        currentAxis = plt.gca()

        p2_image2_Labels = []
        for i in range(top_conf.shape[0]):

            p2_image2_Label = []

            # bbox value
            xmin = int(round(top_xmin[i] * image.shape[1]))
            ymin = int(round(top_ymin[i] * image.shape[0]))
            xmax = int(round(top_xmax[i] * image.shape[1]))
            ymax = int(round(top_ymax[i] * image.shape[0]))
            # score
            score = top_conf[i]
            # label
            label = int(top_label_indices[i])
            if xmin < 0:
                xmin = 0
            if ymin < 0:
                ymin = 0
            if xmax > image.shape[1]:
                xmax = image.shape[1]
            if ymax > image.shape[0]:
                ymax = image.shape[0]
            label_name = top_labels[i]
            box_label.append(str(label_name))
            box2_x1 = xmin + x
            box2_x2 = xmax + x
            box_xmin.append(box2_x1)
            box_ymin.append(ymin)
            box_xmax.append(box2_x2)
            box_ymax.append(ymax)
            box_score.append(score)

            p2_image2_Label.append(save_pic_name)
            p2_image2_Label.append(str(box2_x1))
            p2_image2_Label.append(str(ymin))
            p2_image2_Label.append(str(box2_x2))
            p2_image2_Label.append(str(ymax))
            p2_image2_Label.append(label_name)
            p2_image2_Label.append(str(score))

            p2_image2_Labels.append(p2_image2_Label)

            # WriteinCSV("1.csv", save_pic_name, str(box2_x1), str(ymin), str(box2_x2), str(ymax), label_name,str(score))
        p2_image_i.append(p2_image2_Labels)

        img3 = img.crop((sp[0] - 800, 0, sp[0], sp[1]))
        img3.save(load_path3)
        image = caffe.io.load_image(load_path3)

        # net = caffe.Net(model_def, model_weights, caffe.TEST)
        transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
        transformer.set_transpose('data', (2, 0, 1))
        transformer.set_mean('data', np.array([104, 117, 123]))  # mean pixel
        transformer.set_raw_scale('data',
                                  255)  # the reference model operates on images in [0,255] range instead of [0,1]
        transformer.set_channel_swap('data',
                                     (2, 1, 0))  # the reference model has channels in BGR order instead of RGB
        # image_resize = 512
        # net.blobs['data'].reshape(1, 3, image_resize, image_resize)
        net.blobs['data'].reshape(1, 3, 400, 800)
        transformed_image = transformer.preprocess('data', image)

        # transformed_image = detection_init(labelmap_file, net, image)
        net.blobs['data'].data[...] = transformed_image
        detections = net.forward()['detection_out']

        # Parse the outputs.
        det_label = detections[0, 0, :, 1]
        det_conf = detections[0, 0, :, 2]
        det_xmin = detections[0, 0, :, 3]
        det_ymin = detections[0, 0, :, 4]
        det_xmax = detections[0, 0, :, 5]
        det_ymax = detections[0, 0, :, 6]

        # Get detections with confidence higher than 0.6.
        top_indices = [i for i, conf in enumerate(det_conf) if conf >= score_threshold]
        top_conf = det_conf[top_indices]
        top_label_indices = det_label[top_indices].tolist()

        # get_lablename_fcn
        file = open(labelmap_file, 'r')
        labelmap = caffe_pb2.LabelMap()
        text_format.Merge(str(file.read()), labelmap)
        num_labels = len(labelmap.item)
        labelnames = []

        for label in top_label_indices:
            found = False
            for i in range(0, num_labels):
                if label == labelmap.item[i].label:
                    found = True
                    labelnames.append(labelmap.item[i].display_name)
                    break
            assert found == True

        top_labels = labelnames
        top_xmin = det_xmin[top_indices]
        top_ymin = det_ymin[top_indices]
        top_xmax = det_xmax[top_indices]
        top_ymax = det_ymax[top_indices]
        # Plot the boxes in test picture

        colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
        plt.switch_backend('agg')
        currentAxis = plt.gca()
        p2_image3_Labels = []
        for i in range(top_conf.shape[0]):
            # bbox value

            p2_image3_Label = []

            xmin = int(round(top_xmin[i] * image.shape[1]))
            ymin = int(round(top_ymin[i] * image.shape[0]))
            xmax = int(round(top_xmax[i] * image.shape[1]))
            ymax = int(round(top_ymax[i] * image.shape[0]))
            if xmin < 0:
                xmin = 0
            if ymin < 0:
                ymin = 0
            if xmax > image.shape[1]:
                xmax = image.shape[1]
            if ymax > image.shape[0]:
                ymax = image.shape[0]
            # score
            score = top_conf[i]
            # label
            label = int(top_label_indices[i])
            label_name = top_labels[i]
            box3_x1 = xmin + sp[0] - 800
            box3_x2 = xmax + sp[0] - 800
            box_label.append(str(label_name))
            box_xmin.append(box3_x1)
            box_ymin.append(ymin)
            box_xmax.append(box3_x2)
            box_ymax.append(ymax)
            box_score.append(score)

            p2_image3_Label.append(save_pic_name)
            p2_image3_Label.append(str(box3_x1))
            p2_image3_Label.append(str(ymin))
            p2_image3_Label.append(str(box3_x2))
            p2_image3_Label.append(str(ymax))
            p2_image3_Label.append(label_name)
            p2_image3_Label.append(str(score))

            p2_image3_Labels.append(p2_image3_Label)

            # WriteinCSV("1.csv", save_pic_name, str(box3_x1), str(ymin), str(box3_x2), str(ymax),label_name,str(score))
        p2_image_i.append(p2_image3_Labels)

        box = [box_label, box_xmin, box_ymin, box_xmax, box_ymax, box_score]

        try:
            box_dy, zj, box_s = add_top.main(load_path, box)
            if len(box_dy) > 0:
                p2_image_box = []
                for i in range(len(box_dy)):
                    Lbox = []
                    xmin = box_dy[i][0]
                    ymin = box_dy[i][1]
                    xmax = box_dy[i][2]
                    ymax = box_dy[i][3]
                    label_name = zj[i][0]
                    score = box_s[i][0]

                    Lbox.append(save_pic_name)
                    Lbox.append(str(xmin))
                    Lbox.append(str(ymin))
                    Lbox.append(str(xmax))
                    Lbox.append(str(ymax))
                    Lbox.append(label_name)
                    Lbox.append(str(score))

                    p2_image_box.append(Lbox)
                p2_image_i.append(p2_image_box)
                # WriteinCSV("1.csv", save_pic_name, str(xmin), str(ymin), str(xmax), str(ymax), label_name,str(score))
            else:
                pass
        except TypeError:
            continue
        p2.append(p2_image_i)
    return_list.append(p2)
    p2t = time.time() - p2t1
    print("进程2时间：",p2t)



if __name__ == '__main__':
    z1 = time.time()
    InputImagePath, OutputImagePath, score_threshold, OutputCSVFileName = GetPara()
    InitCSV(OutputCSVFileName)
    jpg_names = GetJPGName(InputImagePath)
    jpgnum = len(jpg_names) / 2
    jpg_names1 = jpg_names[:jpgnum]
    jpg_names2 = jpg_names[jpgnum:]
    manager = Manager()
    return_list = manager.list()

    p1 = Process(target=RunDetection1,args=(InputImagePath, OutputImagePath, jpg_names1, float(score_threshold),return_list))
    p2 = Process(target=RunDetection2,args=(InputImagePath, OutputImagePath, jpg_names2, float(score_threshold),return_list))

    p1.start()
    p2.start()

    p1.join()
    p2.join()
    WriteinCSV('1.csv',return_list)
    for i in os.listdir(InputImagePath):
        ri = os.path.join(InputImagePath, i)
        os.remove(ri)
    zt = time.time()-z1
    print("总时间：",zt)
    



