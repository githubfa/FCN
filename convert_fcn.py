#!/usr/bin/env python3
import logging
import os
import dataset_util
import cv2
import numpy as np
import tensorflow as tf
import dataset_util
from vgg import vgg_16

# os.getcwd()
dirpath = '/home/xiasan/quiz-w9/quiz-w9-code'
os.chdir(dirpath)

flags = tf.app.flags
flags.DEFINE_string('data_dir', '/home/xiasan/quiz-w9/VOCdevkit/VOC2012', 'Root directory to raw pet dataset.')
flags.DEFINE_string('output_dir', '/home/xiasan/quiz-w9', 'Path to directory to output TFRecords.')

FLAGS = flags.FLAGS

classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
           'dog', 'horse', 'motorbike', 'person', 'potted plant',
           'sheep', 'sofa', 'train', 'tv/monitor']
# RGB color for each class
colormap = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
            [128, 0, 128], [0, 128, 128], [
                128, 128, 128], [64, 0, 0], [192, 0, 0],
            [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128],
            [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0],
            [0, 192, 0], [128, 192, 0], [0, 64, 128]]

cm2lbl = np.zeros(256 ** 3)

# m2lbl[2]

for i, cm in enumerate(colormap):
    cm2lbl[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i


def image2label(im):
    data = im.astype('int32')
    # cv2.imread. default channel layout is BGR
    idx = (data[:, :, 2] * 256 + data[:, :, 1]) * 256 + data[:, :, 0]

    return np.array(cm2lbl[idx])


def dict_to_tf_example(data, label):
    filename = os.path.basename(data)

    with open(data, 'rb') as inf:
        encoded_data = inf.read()
    img_label = cv2.imread(label)
    img_mask = image2label(img_label)

    # toBytes()方法是将参数使用UTF-8的编码格式转换成byte[]
    encoded_label = img_mask.astype(np.uint8).tobytes()

    height, width = img_label.shape[0], img_label.shape[1]
    if height < vgg_16.default_image_size or width < vgg_16.default_image_size:
        # 保证最后随机裁剪的尺寸
        return None

    # Your code here, fill the dict
    feature_dict = {
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_data),
        'image/label': dataset_util.bytes_feature(encoded_label),
        'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return example


def create_tf_record(output_filename, file_pars):
    # Your code here

    writer = tf.python_io.TFRecordWriter(output_filename)
    for dir_data, dir_label in file_pars:

        tf_example = dict_to_tf_example(dir_data, dir_label)
        if (tf_example == None):
            continue
        writer.write(tf_example.SerializeToString())

    writer.close()


def read_images_names(root, train=True):
    txt_fname = os.path.join(root, 'ImageSets/Segmentation/', 'train.txt' if train else 'val.txt')

    with open(txt_fname, 'r') as f:
        images = f.read().split()

    data = []
    label = []
    for fname in images:
        data.append('%s/JPEGImages/%s.jpg' % (root, fname))
        label.append('%s/SegmentationClass/%s.png' % (root, fname))
    return zip(data, label)


def main(_):
    logging.info('Prepare dataset file names')

    train_output_path = os.path.join(FLAGS.output_dir, 'fcn_train.record')
    val_output_path = os.path.join(FLAGS.output_dir, 'fcn_val.record')

    train_files = read_images_names(FLAGS.data_dir, True)
    '''
    for a1,b1 in train_files:
      print(a1,b1)
    list(train_files)
    '''
    val_files = read_images_names(FLAGS.data_dir, False)
    create_tf_record(train_output_path, train_files)
    create_tf_record(val_output_path, val_files)


if __name__ == '__main__':
    tf.app.run()
