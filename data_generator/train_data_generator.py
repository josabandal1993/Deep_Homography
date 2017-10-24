#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import os
import cv2
import random
import numpy as np
import shutil
import argparse
import pickle
import time 

iter_max = 79
batch_size = 6400
pairs_per_img = 8  #Chosen 8 so that the total number of unique image is 6400*79/8


dir_train = '/home/jakesabandal/Documents/Deeplearning/Datasets/MSCOCO/train2014'  #change to your own directory

def load_data(raw_data_path):
    dir_list_out = []
    dir_list = os.listdir(raw_data_path)
    if '.' in dir_list:
        dir_list.remove('.')
    if '..' in dir_list:
        dir_list.remove('.')
    if '.DS_Store' in dir_list:
        dir_list.remove('.DS_Store')
    dir_list.sort()
    for i in range(len(dir_list)):
        dir_list_out.append(os.path.join(raw_data_path, dir_list[i]))
    return dir_list_out


def generate_data(img_path):
    data_re = []
    label_re = []
    random_list = []
    img = cv2.resize(cv2.imread(img_path, 0), (320, 240))
    i = 1
    while i < pairs_per_img + 1:
        data = []
        label = []
        y_start = random.randint(32, 80)
        y_end = y_start + 128
        x_start = random.randint(32, 160)
        x_end = x_start + 128

        y_1 = y_start
        x_1 = x_start
        y_2 = y_end
        x_2 = x_start
        y_3 = y_end
        x_3 = x_end
        y_4 = y_start
        x_4 = x_end

        img_patch = img[y_start:y_end, x_start:x_end]  # patch 1

        y_1_offset = random.randint(-32, 32)
        x_1_offset = random.randint(-32, 32)
        y_2_offset = random.randint(-32, 32)
        x_2_offset = random.randint(-32, 32)
        y_3_offset = random.randint(-32, 32)
        x_3_offset = random.randint(-32, 32)
        y_4_offset = random.randint(-32, 32)
        x_4_offset = random.randint(-32, 32)

        y_1_p = y_1 + y_1_offset
        x_1_p = x_1 + x_1_offset
        y_2_p = y_2 + y_2_offset
        x_2_p = x_2 + x_2_offset
        y_3_p = y_3 + y_3_offset
        x_3_p = x_3 + x_3_offset
        y_4_p = y_4 + y_4_offset
        x_4_p = x_4 + x_4_offset

        pts_img_patch = np.array([[y_1,x_1],[y_2,x_2],[y_3,x_3],[y_4,x_4]]).astype(np.float32)
        pts_img_patch_perturb = np.array([[y_1_p,x_1_p],[y_2_p,x_2_p],[y_3_p,x_3_p],[y_4_p,x_4_p]]).astype(np.float32)
        h,status = cv2.findHomography(pts_img_patch, pts_img_patch_perturb, cv2.RANSAC)

        img_perburb = cv2.warpPerspective(img, h, (320, 240))
        img_perburb_patch = img_perburb[y_start:y_end, x_start:x_end]  # patch 2

        if not [y_1,x_1,y_2,x_2,y_3,x_3,y_4,x_4] in random_list:
            data.append(img_patch)
            data.append(img_perburb_patch)  # [2, 128, 128]
            random_list.append([y_1,x_1,y_2,x_2,y_3,x_3,y_4,x_4])
            h_4pt = np.array([y_1_offset,x_1_offset,y_2_offset,x_2_offset,y_3_offset,x_3_offset,y_4_offset,x_4_offset])
            label.append(h_4pt)  # [1, 8]
            i += 1
            data_re.append(data)  # [?, 2, 128, 128]
            label_re.append(label)  # [?, 1, 8]
        
    return data_re, label_re


class DataSet(object):
    def __init__(self, img_path_list):
        self.img_path_list = img_path_list
        self.index_in_epoch = 0
        self.number = len(img_path_list)
        print(int(self.number))

    def next_batch(self):
        start = int(self.index_in_epoch)
        print (start)
        self.index_in_epoch += batch_size / pairs_per_img
        if self.index_in_epoch > self.number:
            self.index_in_epoch = 0
            start = self.index_in_epoch
            self.index_in_epoch += batch_size / pairs_per_img
        end = int(self.index_in_epoch)

        data_batch, label_batch = generate_data(self.img_path_list[start])
        for i in range(start+1, end):

            data, label = generate_data(self.img_path_list[i])
            data_batch = np.concatenate((data_batch, data))
            label_batch = np.concatenate((label_batch, label))

        data_batch =  np.array(data_batch).transpose([0, 2, 3, 1])
        label_batch = np.array(label_batch).squeeze()
        
        return data_batch, label_batch



def main(_):

    train_img_list = load_data(dir_train)
    train_model = DataSet(train_img_list)
    for i in range(iter_max):
        x_train, y_train = train_model.next_batch()
        y_train = np.reshape(y_train, [-1, 8])        
        train = {'features': x_train, 'labels': y_train}
        pickle.dump(train, open("train_"+str(i)+".p", "wb"))
 
        print("Saving in pickle format."+str(i))


if __name__ == "__main__":
    tf.app.run()
