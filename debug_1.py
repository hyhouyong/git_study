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

# the path of caffe & the path of python in caffe
import sys, getopt

sys.path.append('/home/hjin/caffe-ssd/python')
import caffe

caffe_root = '/home/nvidia/caffe-ssd/'

# caffe.set_mode_cpu()
caffe.set_mode_gpu()
caffe.set_device(1)

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


def WriteinCSV(OutputCSVFileName, image_name, xmin, ymin, xmax, ymax, label_name, score):
    WriteCSVFile = open(OutputCSVFileName, 'a')
    one_line_data = [image_name, xmin, ymin, xmax, ymax, label_name, score]
    WriteCSVFile.write(",".join(one_line_data) + "\n")
    WriteCSVFile.close()


def RunDetection(InputImagePath, OutputImagePath, jpg_names, score_threshold):
    net = caffe.Net(model_def, model_weights, caffe.TEST)
    count = 0
    num = 0
    # print('jpg_names:{}'.format(jpg_names))
    for one_jpg in jpg_names:
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
        for i in range(top_conf.shape[0]):
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
            WriteinCSV("box.csv", save_pic_name, str(xmin), str(ymin), str(xmax), str(ymax),
                       label_name,
                       str(score))

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
        for i in range(top_conf.shape[0]):
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
            # WriteinCSV("1.csv", save_pic_name, str(xmin + x), str(ymin), str(xmax + x), str(ymax), label_name,
            #             #            str(score))
            WriteinCSV("box.csv", save_pic_name, str(box2_x1), str(ymin), str(box2_x2), str(ymax), label_name,
                       str(score))

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
        for i in range(top_conf.shape[0]):
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
            box3_x1 = xmin + sp[0] - 800
            box3_x2 = xmax + sp[0] - 800
            box_label.append(str(label_name))
            box_xmin.append(box3_x1)
            box_ymin.append(ymin)
            box_xmax.append(box3_x2)
            box_ymax.append(ymax)
            # WriteinCSV("1.csv", save_pic_name, str(xmin + sp[0] - 800), str(ymin), str(xmax + sp[0] - 800), str(ymax),
            #            label_name,
            #            str(score))
            WriteinCSV("box.csv", save_pic_name, str(box3_x1), str(ymin), str(box3_x2), str(ymax),
                       label_name,
                       str(score))
        box = [box_label, box_xmin, box_ymin, box_xmax, box_ymax]
        try:
            box_dy, zj, zj_score = add_top.main(load_path, box)
	    print( save_pic_name)
            image = cv2.imread(load_path)
            if len(box_dy) > 0:
                for i in range(len(box_dy)):
                    xmin = box_dy[i][0]
                    ymin = box_dy[i][1]
                    xmax = box_dy[i][2]
                    ymax = box_dy[i][3]
                    label_name = zj[i][0]
                    score = zj_score[i][0]
                    cv2.rectangle(image, (xmin, ymin),
                                  (xmax, ymax),
                                  (0, 255, 0),
                                  1)
                    cv2.putText(image, label_name + "_" + score, (xmax, ymax + 20), cv2.FONT_HERSHEY_COMPLEX, 0.4,
                                (0, 255, 0),
                                1)
                    cv2.imwrite(save_path, image)
                    WriteinCSV("box.csv", save_pic_name, str(xmin), str(ymin), str(xmax), str(ymax), label_name,
                               str(score))

            else:
                pass
            print(zj_score)
            score_list = []
            for score in zj_score:
                score_list.append(float(score[0]))
            if len(zj_score) != 0:
                if max(score_list) > 0.7:
                    with open("no.txt", 'a') as f:
                        for i in range(len(box_dy)):
                            xmin = box_dy[i][0]
                            ymin = box_dy[i][1]
                            xmax = box_dy[i][2]
                            ymax = box_dy[i][3]
                            label_name = zj[i][0]
                            score = zj_score[i][0]
                            f.write(
                                str(save_pic_name) + " " + str(xmin) + " " + str(ymin) + " " + str(xmax) + " " + str(
                                    ymax) + " " + str(
                                    label_name) + " " + str(score) + "\n")
                        count += 1
        except TypeError:
            continue
        num += 1
    return num, count


if __name__ == '__main__':
    InputImagePath, OutputImagePath, score_threshold, OutputCSVFileName = GetPara()
    InitCSV(OutputCSVFileName)
    jpg_names = GetJPGName(InputImagePath)
    num, count = RunDetection(InputImagePath, OutputImagePath, jpg_names, float(score_threshold))
    print("sum", num)
    print("no", count)
