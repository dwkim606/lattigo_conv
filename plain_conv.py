# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import numpy as np
import time
import sys
import math
import cmath
import random
import os
import tensorflow as tf
from tensorflow.keras.layers import Cropping2D

def prt_list(input, start, width, showAll):
    if showAll:
        for i in range(len(input) // width):
            if (i%(width) == 0):
                print()
            print(input[start + i * width: start + i * width + width])
    else:
        for i in range(width):
            print(input[start + i * width: start + i * width + width])

def plain_imagenet(input_image, ker_name):
    in_dir = 'weight_imgnet_'+ker_name+'h5/'
    in_wid = [112, 56, 28, 14, 7]
    batch = [64, 64, 128, 256, 512]
    
    ker_list = {'ker3_': 3, 'ker5_': 5, 'ker7_': 7}
    blc_list = {'ker3_': [1,4,4,4,4], 'ker5_': [1,4,4,2,2], 'ker7_': [1,4,4,1,1]}

    ker_wid = ker_list[ker_name]
    blcs = blc_list[ker_name] #ker7 [3,2,2] #ker5 [5,4,4] #ker3 [7,6,6]

    print(input_image)
    
    num = 0            
    for blc in range(5):
        if blc == 0: # 7*7 64, stride 2
            ten_k = tf.reshape(tf.constant(np.loadtxt(in_dir+'w'+str(num)+'-conv.csv'), tf.float32), [7, 7, 3, batch[blc]])
            conv = tf.nn.conv2d(input_image, ten_k, strides=[1,2,2,1], padding="SAME")
            num += 1
        else:
            if blc == 1: # max pool
                conv = tf.nn.max_pool(conv, ksize=3, strides=2, padding="SAME")
            for i in range(blcs[blc]):
                if (blc != 1) and (i == 0):
                    bn_a = [[np.loadtxt(in_dir+'w'+str(num)+'-a.csv') for i in range(in_wid[blc-1])] for j in range(in_wid[blc-1])]
                    bn_b = [[np.loadtxt(in_dir+'w'+str(num)+'-b.csv') for i in range(in_wid[blc-1])] for j in range(in_wid[blc-1])]
                    ten_a = tf.reshape(tf.constant(bn_a, tf.float32), [1, in_wid[blc-1], in_wid[blc-1], batch[blc-1]])
                    ten_b = tf.reshape(tf.constant(bn_b, tf.float32), [1, in_wid[blc-1], in_wid[blc-1], batch[blc-1]])
                    conv = ten_a * conv + ten_b
                    conv = tf.nn.relu(conv)

                    ten_k = tf.reshape(tf.constant(np.loadtxt(in_dir+'w'+str(num)+'-conv.csv'), tf.float32), [ker_wid, ker_wid, batch[blc-1], batch[blc]])
                    conv = tf.nn.conv2d(conv, ten_k, strides = [1,2,2,1], padding = "SAME")
                else:
                    bn_a = [[np.loadtxt(in_dir+'w'+str(num)+'-a.csv') for i in range(in_wid[blc])] for j in range(in_wid[blc])]
                    bn_b = [[np.loadtxt(in_dir+'w'+str(num)+'-b.csv') for i in range(in_wid[blc])] for j in range(in_wid[blc])]
                    ten_a = tf.reshape(tf.constant(bn_a, tf.float32), [1, in_wid[blc], in_wid[blc], batch[blc]])
                    ten_b = tf.reshape(tf.constant(bn_b, tf.float32), [1, in_wid[blc], in_wid[blc], batch[blc]])
                    conv = ten_a * conv + ten_b
                    conv = tf.nn.relu(conv)

                    ten_k = tf.reshape(tf.constant(np.loadtxt(in_dir+'w'+str(num)+'-conv.csv'), tf.float32), [ker_wid, ker_wid, batch[blc], batch[blc]])
                    conv = tf.nn.conv2d(conv, ten_k, strides = [1,1,1,1], padding = "SAME")
                num +=1
                if i == 0:
                    print(blc," to ", blc+1," block. ")
                    print(conv)

    bn_a = [[np.loadtxt(in_dir+'w'+str(num)+'-a.csv') for i in range(in_wid[blc])] for j in range(in_wid[blc])]
    bn_b = [[np.loadtxt(in_dir+'w'+str(num)+'-b.csv') for i in range(in_wid[blc])] for j in range(in_wid[blc])]
    ten_a = tf.reshape(tf.constant(bn_a, tf.float32), [1, in_wid[blc], in_wid[blc], batch[blc]])
    ten_b = tf.reshape(tf.constant(bn_b, tf.float32), [1, in_wid[blc], in_wid[blc], batch[blc]])
    ten_final = tf.reshape(tf.constant(np.loadtxt(in_dir+'fc.csv'), tf.float32), [1, 1, 512, 1000])
    
    conv = ten_a * conv + ten_b
    conv = tf.nn.relu(conv)
    conv = tf.reduce_mean(conv, [1,2], keepdims = True)
    conv = tf.nn.conv2d(conv, ten_final, strides = [1,1,1,1], padding = "SAME")

    conv = tf.squeeze(conv, axis=[1,2])
    print(conv)
    # conv = tf.argmax(conv, 1)
    return conv

# if prec = false, take mid_input = ct_in
def plain_imagenet_after_mid(mid_input, ker_name, mid_input_prec):
    in_dir = 'weight_imgnet_'+ker_name+'h5/'
    in_wid = [28, 14, 7]
    batch = [128, 256, 512]
    
    ker_list = {'ker3_': 3, 'ker5_': 5, 'ker7_': 7}
    blc_list = {'ker3_': [4,4], 'ker5_': [2,2], 'ker7_': [1,1]}

    ker_wid = ker_list[ker_name]
    blcs = blc_list[ker_name] #ker7 [3,2,2] #ker5 [5,4,4] #ker3 [7,6,6]

    num = 9
    if mid_input_prec:
        bn_a = [[np.loadtxt(in_dir+'w'+str(num)+'-a.csv') for i in range(in_wid[0])] for j in range(in_wid[0])]
        bn_b = [[np.loadtxt(in_dir+'w'+str(num)+'-b.csv') for i in range(in_wid[0])] for j in range(in_wid[0])]
        ten_a = tf.reshape(tf.constant(bn_a, tf.float32), [1, in_wid[0], in_wid[0], batch[0]])
        ten_b = tf.reshape(tf.constant(bn_b, tf.float32), [1, in_wid[0], in_wid[0], batch[0]])
        eval_start = tf.nn.relu(ten_a * mid_input + ten_b)

        count = 0
        while os.path.exists('Imagenet/ker3_ct_in/ker3_ct_in'+str(count)+'.csv'):
            count += 1
        np.savetxt('Imagenet/ker3_ct_in/ker3_ct_in'+str(count)+'.csv', tf.reshape(eval_start, [-1]).numpy())
        conv = eval_start
    else:
        conv = mid_input

    
    for i in range(blcs[0]):
        if i == 0:
            ten_k = tf.reshape(tf.constant(np.loadtxt(in_dir+'w'+str(num)+'-conv.csv'), tf.float32), [ker_wid, ker_wid, batch[0], batch[1]])
            conv = tf.nn.conv2d(conv, ten_k, strides = [1,2,2,1], padding = "SAME")
        else:
            ten_k = tf.reshape(tf.constant(np.loadtxt(in_dir+'w'+str(num)+'-conv.csv'), tf.float32), [ker_wid, ker_wid, batch[1], batch[1]])
            conv = tf.nn.conv2d(conv, ten_k, strides = [1,1,1,1], padding = "SAME")
        num+= 1
        bn_a = [[np.loadtxt(in_dir+'w'+str(num)+'-a.csv') for i in range(in_wid[1])] for j in range(in_wid[1])]
        bn_b = [[np.loadtxt(in_dir+'w'+str(num)+'-b.csv') for i in range(in_wid[1])] for j in range(in_wid[1])]
        ten_a = tf.reshape(tf.constant(bn_a, tf.float32), [1, in_wid[1], in_wid[1], batch[1]])
        ten_b = tf.reshape(tf.constant(bn_b, tf.float32), [1, in_wid[1], in_wid[1], batch[1]])
        # print(tf.reduce_max(ten_a*conv+ten_b, [0,1,2,3]))
        # print(tf.reduce_mean(ten_a*conv+ten_b, [0,1,2,3]))
        conv = tf.nn.relu(ten_a * conv + ten_b)
    
    count = 0
    while os.path.exists('Imagenet/ker3_ct_in_2/ker3_ct_in'+str(count)+'.csv'):
        count += 1
    np.savetxt('Imagenet/ker3_ct_in_2/ker3_ct_in'+str(count)+'.csv', tf.reshape(conv, [-1]).numpy())

    for i in range(blcs[1]):
        if i == 0:
            ten_k = tf.reshape(tf.constant(np.loadtxt(in_dir+'w'+str(num)+'-conv.csv'), tf.float32), [ker_wid, ker_wid, batch[1], batch[2]])
            conv = tf.nn.conv2d(conv, ten_k, strides = [1,2,2,1], padding = "SAME")
        else:
            ten_k = tf.reshape(tf.constant(np.loadtxt(in_dir+'w'+str(num)+'-conv.csv'), tf.float32), [ker_wid, ker_wid, batch[2], batch[2]])
            conv = tf.nn.conv2d(conv, ten_k, strides = [1,1,1,1], padding = "SAME")
        num+= 1
        bn_a = [[np.loadtxt(in_dir+'w'+str(num)+'-a.csv') for i in range(in_wid[2])] for j in range(in_wid[2])]
        bn_b = [[np.loadtxt(in_dir+'w'+str(num)+'-b.csv') for i in range(in_wid[2])] for j in range(in_wid[2])]
        ten_a = tf.reshape(tf.constant(bn_a, tf.float32), [1, in_wid[2], in_wid[2], batch[2]])
        ten_b = tf.reshape(tf.constant(bn_b, tf.float32), [1, in_wid[2], in_wid[2], batch[2]])
        # print(tf.reduce_max(ten_a*conv+ten_b, [0,1,2,3]))
        # print(tf.reduce_mean(ten_a*conv+ten_b, [0,1,2,3]))
        conv = tf.nn.relu(ten_a * conv + ten_b)

    ten_final = tf.reshape(tf.constant(np.loadtxt(in_dir+'fc.csv'), tf.float32), [1, 1, batch[2], 1000])    
    conv = tf.reduce_mean(conv, [1,2], keepdims = True)
    conv = tf.nn.conv2d(conv, ten_final, strides = [1,1,1,1], padding = "SAME")

    conv = tf.squeeze(conv, axis=[1,2])
    # conv = tf.argmax(conv, 1)
    return conv


def plain_imagenet_bench():
    input_width = 28
    num_bl1 = 1
    num_bl2 = 1
    batch = 8
    vec_size = batch*input_width**2
    ker_width = 3
    bn_a = [0.2, 0.2, 0.2]

    batch1 = batch*2
    batch2 = batch1*2
    ker_size = ker_width**2

    print("input width:", input_width)
    print("batch:", batch)
    print("ker width:", ker_width)
    #data = np.loadtxt("./weights/batch0.txt")

    ## Correctness Check: Compare with TF NN CONV2D
    raw_input = [1.0  for i in range(vec_size)] #[1.0 - (1.0*i)/vec_size for i in range(vec_size)]
    ker01 = [0.1 for i in range(batch * batch1 * ker_size)] #[(0.1 * i)/(batch * batch1 * ker_size) for i in range(batch * batch1 * ker_size)] #[0.1 * i / (batch * batch * filter_size) for i in range(batch * batch * filter_size)]
    ker1 =  [(0.1 * i)/(batch1 * batch1 * ker_size) for i in range(batch1 * batch1 * ker_size)] #[0.1 * i / (batch * batch * filter_size) for i in range(batch * batch * filter_size)]
    ker12 = [(0.1 * i)/(batch1 * batch2 * ker_size) for i in range(batch1 * batch2 * ker_size)] #[0.1 * i / (batch * batch * filter_size) for i in range(batch * batch * filter_size)]
    ker2 = [(0.1 * i)/(batch2 * batch2 * ker_size) for i in range(batch2 * batch2 * ker_size)] #[0.1 * i / (batch * batch * filter_size) for i in range(batch * batch * filter_size)]
    final = [1.0*i/(batch2*60) for i in range(batch2 * 60)]
    bias = [0.0 for i in range(60)]

    ten_x = tf.reshape(tf.constant(np.array(raw_input), tf.float32), [1, input_width, input_width, batch])
    ten_k01 = tf.reshape(tf.constant(np.array(ker01), tf.float32), [ker_width, ker_width, batch, batch1])
    ten_k1 = tf.reshape(tf.constant(np.array(ker1), tf.float32), [ker_width, ker_width, batch1, batch1])
    ten_k12 = tf.reshape(tf.constant(np.array(ker12), tf.float32), [ker_width, ker_width, batch1, batch2])
    ten_k2 = tf.reshape(tf.constant(np.array(ker2), tf.float32), [ker_width, ker_width, batch2, batch2])
    ten_final = tf.reshape(tf.constant(np.array(final), tf.float32), [1, 1, batch2, 60])
    ten_bias = tf.reshape(tf.constant(np.array(bias), tf.float32), [60])

    conv = ten_x
    print("Input: ", conv)
    conv = tf.nn.conv2d(conv, ten_k01, strides = [1,2,2,1], padding = "SAME")*bn_a[0]
    conv = tf.nn.relu(conv)
    print("After 0 to 1 block\n", conv, "\n")

    for i in range(num_bl1):
        conv = tf.nn.conv2d(conv, ten_k1, strides = [1,1,1,1], padding = "SAME")*bn_a[0]
        conv = tf.nn.relu(conv)
        print(i+1,"layer done\n")
    print("after 1st block\n", conv, "\n")

    conv = tf.nn.conv2d(conv, ten_k12, strides = [1,2,2,1], padding = "SAME")*bn_a[1]
    conv = tf.nn.relu(conv)
    print("after 1st to 2nd block\n", conv, "\n")

    for i in range(num_bl2):
        conv = tf.nn.conv2d(conv, ten_k2, strides = [1,1,1,1], padding = "SAME")*bn_a[2]
        conv = tf.nn.relu(conv)
        print(i+1,"layer done\n")
    print("after 2nd block\n", conv, "\n")

    conv = tf.reduce_mean(conv, [1,2], keepdims = True)
    conv = tf.nn.conv2d(conv, ten_final, strides = [1,1,1,1], padding = "SAME")
    conv = conv + ten_bias
    conv = tf.squeeze(conv, axis=[1,2])

    print(conv)


# only perform final block
def plain_imagenet_final_bench():
    input_width = 14
    num_blc = 3
    batch = 16
    vec_size = batch*input_width**2
    ker_width = 3
    bn_a = [0.2, 0.2]

    batch2 = batch*2
    ker_size = ker_width**2

    print("input width:", input_width)
    print("batch:", batch)
    print("ker width:", ker_width)
    #data = np.loadtxt("./weights/batch0.txt")

    ## Correctness Check: Compare with TF NN CONV2D
    raw_input = [1.0 - (1.0*i)/vec_size for i in range(vec_size)]
    ker12 = [(0.1 * i)/(batch * batch2 * ker_size) for i in range(batch * batch2 * ker_size)] #[0.1 * i / (batch * batch * filter_size) for i in range(batch * batch * filter_size)]
    ker2 = [(0.1 * i)/(batch2 * batch2 * ker_size) for i in range(batch2 * batch2 * ker_size)] #[0.1 * i / (batch * batch * filter_size) for i in range(batch * batch * filter_size)]
    final = [1.0*i/(batch2*60) for i in range(batch2 * 60)]
    bias = [0.0 for i in range(60)]

    ten_x = tf.reshape(tf.constant(np.array(raw_input), tf.float32), [1, input_width, input_width, batch])
    ten_k12 = tf.reshape(tf.constant(np.array(ker12), tf.float32), [ker_width, ker_width, batch, batch2])
    ten_k2 = tf.reshape(tf.constant(np.array(ker2), tf.float32), [ker_width, ker_width, batch2, batch2])
    ten_final = tf.reshape(tf.constant(np.array(final), tf.float32), [1, 1, batch2, 60])
    ten_bias = tf.reshape(tf.constant(np.array(bias), tf.float32), [60])

    conv = ten_x
    print("Input: ", conv)
    conv = tf.nn.conv2d(conv, ten_k12, strides = [1,2,2,1], padding = "SAME")*bn_a[0]
    conv = tf.nn.relu(conv)
    print("after 1st to 2nd block\n", conv, "\n")

    for i in range(num_blc):
        conv = tf.nn.conv2d(conv, ten_k2, strides = [1,1,1,1], padding = "SAME")*bn_a[1]
        conv = tf.nn.relu(conv)
        print(i+1,"layer done\n")
    print("after 2nd block\n", conv, "\n")

    conv = tf.reduce_mean(conv, [1,2], keepdims = True)
    conv = tf.nn.conv2d(conv, ten_final, strides = [1,1,1,1], padding = "SAME")
    conv = conv + ten_bias
    conv = tf.squeeze(conv, axis=[1,2])

    print(conv)


def plain_resnet_bench():
    input_width = 32
    batch = 4
    vec_size = 3*input_width**2
    ker_width = 3
    bn_a = [0.2, 0.2, 0.2]

    num_bl1 = 2
    num_bl2 = 2
    num_bl3 = 2

    batch2 = batch*2
    batch3 = batch2*2
    ker_size = ker_width**2

    print("input width:", input_width)
    print("batch:", batch)
    print("ker width:", ker_width)
    #data = np.loadtxt("./weights/batch0.txt")

    ## Correctness Check: Compare with TF NN CONV2D
    raw_input = [1.0-(1.0 * i)/vec_size for i in range(vec_size)]
    ker0 = [(0.25 * i)/(3 * batch * ker_size) for i in range(3 * batch * ker_size)] #[0.1 * i / (batch * batch * filter_size) for i in range(batch * batch * filter_size)]
    ker =  [(0.25 * i)/(batch * batch * ker_size) for i in range(batch * batch * ker_size)] #[0.1 * i / (batch * batch * filter_size) for i in range(batch * batch * filter_size)]
    ker12 = [(0.25 * i)/(batch * batch2 * ker_size) for i in range(batch * batch2 * ker_size)] #[0.1 * i / (batch * batch * filter_size) for i in range(batch * batch * filter_size)]
    ker2 = [(0.25 * i)/(batch2 * batch2 * ker_size) for i in range(batch2 * batch2 * ker_size)] #[0.1 * i / (batch * batch * filter_size) for i in range(batch * batch * filter_size)]
    ker23 = [(0.25 * 100)/(batch2 * batch3 * ker_size) for i in range(batch2 * batch3 * ker_size)]  # (0.5 * i)/(batch2 * batch3 * ker_size) #[0.1 * i / (batch * batch * filter_size) for i in range(batch * batch * filter_size)]
    ker3 = [(0.25 * i)/(batch3 * batch3 * ker_size) for i in range(batch3 * batch3 * ker_size)] #[0.1 * i / (batch * batch * filter_size) for i in range(batch * batch * filter_size)]

    ten_x = tf.reshape(tf.constant(np.array(raw_input), tf.float32), [1, input_width, input_width, 3])
    ten_k0 = tf.reshape(tf.constant(np.array(ker0), tf.float32), [ker_width, ker_width, 3, batch])
    ten_k = tf.reshape(tf.constant(np.array(ker), tf.float32), [ker_width, ker_width, batch, batch])
    ten_k12 = tf.reshape(tf.constant(np.array(ker12), tf.float32), [ker_width, ker_width, batch, batch2])
    ten_k2 = tf.reshape(tf.constant(np.array(ker2), tf.float32), [ker_width, ker_width, batch2, batch2])
    ten_k23 = tf.reshape(tf.constant(np.array(ker23), tf.float32), [ker_width, ker_width, batch2, batch3])
    ten_k3 = tf.reshape(tf.constant(np.array(ker3), tf.float32), [ker_width, ker_width, batch3, batch3])
    # print(ten_k)

    # conv = tf.nn.conv2d_transpose(ten_x, ten_k1, output_shape=(1, input_width[1], input_width[1], batch[1]), strides=[1, 2, 2, 1])

    # conv = tf.reshape(tf.constant(np.array(mid_input), tf.float32), [1, 16, 16, batch2])
    # print("Input: ", conv)
    
    conv = ten_x
    print("Input: ", conv)
    for i in range(num_bl1):
        if i == 0:
            conv = tf.nn.conv2d(conv, ten_k0, strides = [1,1,1,1], padding = "SAME")*bn_a[0]    
        else:
            conv = tf.nn.conv2d(conv, ten_k, strides = [1,1,1,1], padding = "SAME")*bn_a[0]
        conv = tf.nn.relu(conv)
        print(conv)
        print(i+1,"layer done\n")
    print("after 1st block\n", conv, "\n")

    conv = tf.nn.conv2d(conv, ten_k12, strides = [1,2,2,1], padding = "SAME")*bn_a[1]
    conv = tf.nn.relu(conv)
    print("after 1st to 2nd block\n", conv, "\n")

    for i in range(num_bl2):
        conv = tf.nn.conv2d(conv, ten_k2, strides = [1,1,1,1], padding = "SAME")*bn_a[1]
        conv = tf.nn.relu(conv)
        print(conv)
        print(i+1,"layer done\n")
    print("after 2nd block\n", conv, "\n")

    # tmp_input = [0.01 * i for i in range(4*4*4)]
    # ten_tmp_x = tf.reshape(tf.constant(np.array(tmp_input), tf.float32), [1, 4, 4, 4])
    # tmp_ker23 = [0.01*i for i in range(4 * 2 * ker_size)] #[0.1 * i / (batch * batch * filter_size) for i in range(batch * batch * filter_size)]
    # ten_tmp_k23 = tf.reshape(tf.constant(np.array(tmp_ker23), tf.float32), [ker_width, ker_width, 4, 2])
    # conv = tf.nn.conv2d(ten_tmp_x, ten_tmp_k23, strides = [1,2,2,1], padding = "SAME")*bn_a[2]

    conv = tf.nn.conv2d(conv, ten_k23, strides = [1,2,2,1], padding = "SAME")*bn_a[2]
    conv = tf.nn.relu(conv)
    print("after 2nd to 3rd block\n", conv, "\n")

    for i in range(num_bl3):
        conv = tf.nn.conv2d(conv, ten_k3, strides = [1,1,1,1], padding = "SAME")*bn_a[2]
        conv = tf.nn.relu(conv)
        print(conv)
        print(i+1,"layer done\n")
    print("after 3rd block\n", conv, "\n")

def conv_bnReLU_BL_bench(trans, strides, relu):
    batch = 4
    input_width = 14
    vec_size = batch*input_width**2
    ker_width = 3
    bn_a = 1.0
    ker_size = ker_width**2 
    if trans:
        out_batch = batch//4 
    elif strides:
        out_batch = batch
    else:
        out_batch = batch
    ker_len = batch * out_batch * ker_size

    ## Correctness Check: Compare with TF NN CONV2D
    raw_input = [1.0*i/vec_size for i in range(vec_size)]  # #[1.0*i for i in range(vec_size)] 
    ker = [1.0*i/ker_len for i in range(ker_len)] #  #[1.0 for i in range(batch * out_batch * ker_size)] #[0.1 * i / (batch * batch * filter_size) for i in range(batch * batch * filter_size)]

    print("input width:", input_width)
    print("batch:", batch)
    print("ker width:", ker_width)
    #data = np.loadtxt("./weights/batch0.txt")

    ten_x = tf.reshape(tf.constant(np.array(raw_input), tf.float32), [1, input_width, input_width, batch])
    print("\n input: \n", ten_x)
    if trans:
        ten_k = tf.reshape(tf.constant(np.array(ker), tf.float32), [ker_width, ker_width, out_batch, batch])
        conv = tf.nn.conv2d_transpose(ten_x, ten_k, output_shape=(1, 2*input_width, 2*input_width, out_batch), strides=[1, 2, 2, 1], padding="SAME")*bn_a
    elif strides:
        ten_k = tf.reshape(tf.constant(np.array(ker), tf.float32), [ker_width, ker_width, batch, out_batch])
        conv = tf.nn.conv2d(ten_x, ten_k, strides = [1,2,2,1], padding = "SAME")*bn_a
    else:
        ten_k = tf.reshape(tf.constant(np.array(ker), tf.float32), [ker_width, ker_width, batch, out_batch])
        conv = tf.nn.conv2d(ten_x, ten_k, strides = [1,1,1,1], padding = "SAME")*bn_a
    
    if relu:
        conv = tf.nn.relu(conv)
    
    print("\n result: \n", conv, "\n")

def trans_conv_bnReLU_BL_bench():
    input_width = 2
    batch = 4
    out_batch = batch//4
    vec_size = batch*input_width**2
    ker_width = 7
    bn_a = 1.0

    ker_size = ker_width**2 

    print("input width:", input_width)
    print("batch:", batch)
    print("ker width:", ker_width)
    #data = np.loadtxt("./weights/batch0.txt")

    ## Correctness Check: Compare with TF NN CONV2D
    raw_input = [1.0*i/vec_size for i in range(vec_size)]
    ker =  [1.0 - 1.0*i /(batch*out_batch*ker_size) for i in range(batch * out_batch * ker_size)] #[0.1 * i / (batch * batch * filter_size) for i in range(batch * batch * filter_size)]

    ten_x = tf.reshape(tf.constant(np.array(raw_input), tf.float32), [1, input_width, input_width, batch])
    print("input: \n", ten_x)
    ten_k = tf.reshape(tf.constant(np.array(ker), tf.float32), [ker_width, ker_width, out_batch, batch])

    # print(tf.reshape(ten_k, [-1]))
    # print("kernel: \n", ten_k)

    # print("ker0: ", ten_k[:,:,0,0])
    conv = tf.nn.conv2d_transpose(ten_x, ten_k, output_shape=(1, 2*input_width, 2*input_width, out_batch), strides=[1, 2, 2, 1], padding="SAME")
    # conv = tf.nn.relu(conv)
    print("result: \n", conv, "\n")

def plain_resnet(input_image, ker_name):
    in_dir = 'weight_'+ker_name+'h5/'
    in_wid = [32, 16, 8]
    batch = [16, 32, 64]
    
    ker_list = {'ker3_': 3, 'ker5_': 5, 'ker7_': 7}
    blc_list = {'ker3_': [7,6,6], 'ker5_': [5,4,4], 'ker7_': [3,2,2]}

    ker_wid = ker_list[ker_name]
    ker_size = ker_wid**2
    
    # load weights
    # blocks = [1, 2, 3]
    # units = range(2)
    # kinds = ['a', 'b', 'conv']

    # raw_input = [0.1*i/3000.0 for i in range(32*32*3)]
    # conv = tf.reshape(tf.constant(np.array(raw_input), tf.float32), [1, in_wid[0], in_wid[0], 3])
    conv = input_image
    blcs = blc_list[ker_name] #ker7 [3,2,2] #ker5 [5,4,4] #ker3 [7,6,6]

    # print("Input: ", conv)

    num = 0
    print(input_image)
    for blc in range(3):
        for i in range(blcs[blc]):
            if i == 0:
                if blc == 0:
                    ten_k = tf.reshape(tf.constant(np.loadtxt(in_dir+'w'+str(num)+'-conv.csv'), tf.float32), [ker_wid, ker_wid, 3, batch[blc]])
                    conv = tf.nn.conv2d(conv, ten_k, strides = [1,1,1,1], padding = "SAME")
                else:
                    ten_k = tf.reshape(tf.constant(np.loadtxt(in_dir+'w'+str(num)+'-conv.csv'), tf.float32), [ker_wid, ker_wid, batch[blc-1], batch[blc]])
                    conv = tf.nn.conv2d(conv, ten_k, strides = [1,2,2,1], padding = "SAME")
            else:
                ten_k = tf.reshape(tf.constant(np.loadtxt(in_dir+'w'+str(num)+'-conv.csv'), tf.float32), [ker_wid, ker_wid, batch[blc], batch[blc]])
                conv = tf.nn.conv2d(conv, ten_k, strides = [1,1,1,1], padding = "SAME")
            bn_a = [[np.loadtxt(in_dir+'w'+str(num)+'-a.csv') for i in range(in_wid[blc])] for j in range(in_wid[blc])]
            bn_b = [[np.loadtxt(in_dir+'w'+str(num)+'-b.csv') for i in range(in_wid[blc])] for j in range(in_wid[blc])]
            ten_a = tf.reshape(tf.constant(bn_a, tf.float32), [1, in_wid[blc], in_wid[blc], batch[blc]])
            ten_b = tf.reshape(tf.constant(bn_b, tf.float32), [1, in_wid[blc], in_wid[blc], batch[blc]])
            conv = ten_a * conv + ten_b
            conv = tf.nn.relu(conv)
            if i == 0:
                print(blc," to ", blc+1," block. ")
                # print(conv)
            else:
                print(blc+1,"-th block, ", i,"-th layer")
                # print(conv)
            num += 1
            # print(i+1,"layer done\n")
        # print("after", blc_iter+1, "-th block\n", conv, "\n")

    ten_final = tf.reshape(tf.constant(np.loadtxt(in_dir+'final-fckernel.csv'), tf.float32), [1, 1, 64, 10])
    bias_final = tf.reshape(tf.constant(np.loadtxt(in_dir+'final-fcbias.csv'), tf.float32), [10])
    conv = tf.reduce_mean(conv, [1,2], keepdims = True)
    conv = tf.nn.conv2d(conv, ten_final, strides = [1,1,1,1], padding = "SAME")
    conv = conv + bias_final

    conv = tf.squeeze(conv, axis=[1,2])
    # conv = tf.argmax(conv, 1)
    return conv

def plain_fast_resnet(input_image, ker_name):
    in_dir = 'weight_'+ker_name+'h5/'
    in_wid = [32, 16, 8]
    batch = [16, 32, 64]
    
    ker_list = {'ker3_': 3, 'ker5_': 5, 'ker7_': 7}
    blc_list = {'ker3_': [7,6,6], 'ker5_': [5,4,4], 'ker7_': [3,2,2]}
    pad_list = {'ker3_': 1, 'ker5_': 2, 'ker7_': 3}

    ker_wid = ker_list[ker_name]
    ker_size = ker_wid**2
    pad_size = pad_list[ker_name]
    
    # pad = [[[1.0 if ((i<32-pad_size) and (j<32-pad_size)) else 0.0 for k in range(3)] for i in range(32)] for j in range(32)] 
    # pad0 = [[[1.0 if ((i<32-pad_size) and (j<32-pad_size)) else 0.0 for k in range(16)] for i in range(32)] for j in range(32)] 
    # pad1 = [[[1.0 if ((i<16-pad_size) and (j<16-pad_size)) else 0.0 for k in range(32)] for i in range(16)] for j in range(16)] 
    # pad2 = [[[1.0 if ((i<8-pad_size) and (j<8-pad_size)) else 0.0 for k in range(64)] for i in range(8)] for j in range(8)] 
    pad_size = 2
    pad = [[[1.0 if ((0<i)and(i<32-pad_size) and (0<j)and (j<32-pad_size)) else 0.0 for k in range(3)] for i in range(32)] for j in range(32)] 
    pad0 = [[[1.0 if ((0<i)and(i<32-pad_size) and (0<j)and (j<32-pad_size)) else 0.0 for k in range(16)] for i in range(32)] for j in range(32)] 
    pad1 = [[[1.0 if ((0<i)and(i<16-pad_size) and (0<j)and(j<16-pad_size)) else 0.0 for k in range(32)] for i in range(16)] for j in range(16)] 
    pad2 = [[[1.0 if ((0<i)and(i<8-pad_size) and (0<j)and(j<8-pad_size)) else 0.0 for k in range(64)] for i in range(8)] for j in range(8)] 

    pads = {-1: pad, 0: pad0, 1:pad1, 2:pad2}

    conv = input_image
    # conv = tf.pad(Cropping2D(cropping=((0,1),(0,1)))(input_image), tf.constant([[],[0,1],[0,1],[]])  ,"CONSTANT")
    blcs = blc_list[ker_name] #ker7 [3,2,2] #ker5 [5,4,4] #ker3 [7,6,6]

    # print("Input: ", conv)

    num = 0
    # print(input_image)
    for blc in range(3):
        for i in range(blcs[blc]):
            if i == 0:
                if blc == 0:
                    ten_k = tf.reshape(tf.constant(np.loadtxt(in_dir+'w'+str(num)+'-conv.csv'), tf.float32), [ker_wid, ker_wid, 3, batch[blc]])
                    ten_pad = tf.reshape(tf.constant(pads[-1], tf.float32), [1, 32, 32, 3])
                    conv = conv * ten_pad
                    # print(conv)
                    conv = tf.nn.conv2d(conv, ten_k, strides = [1,1,1,1], padding = "SAME")
                else:
                    ten_k = tf.reshape(tf.constant(np.loadtxt(in_dir+'w'+str(num)+'-conv.csv'), tf.float32), [ker_wid, ker_wid, batch[blc-1], batch[blc]])
                    ten_pad = tf.reshape(tf.constant(pads[blc-1], tf.float32), [1, in_wid[blc-1], in_wid[blc-1], batch[blc-1]])
                    conv = conv*ten_pad
                    # print(conv)
                    conv = tf.nn.conv2d(conv, ten_k, strides = [1,2,2,1], padding = "SAME")
            else:
                ten_k = tf.reshape(tf.constant(np.loadtxt(in_dir+'w'+str(num)+'-conv.csv'), tf.float32), [ker_wid, ker_wid, batch[blc], batch[blc]])
                ten_pad = tf.reshape(tf.constant(pads[blc], tf.float32), [1, in_wid[blc], in_wid[blc], batch[blc]])
                conv = conv * ten_pad
                # print(conv)
                conv = tf.nn.conv2d(conv, ten_k, strides = [1,1,1,1], padding = "SAME")
            bn_a = [[np.loadtxt(in_dir+'w'+str(num)+'-a.csv') for i in range(in_wid[blc])] for j in range(in_wid[blc])]
            bn_b = [[np.loadtxt(in_dir+'w'+str(num)+'-b.csv') for i in range(in_wid[blc])] for j in range(in_wid[blc])]
            ten_a = tf.reshape(tf.constant(bn_a, tf.float32), [1, in_wid[blc], in_wid[blc], batch[blc]])
            ten_b = tf.reshape(tf.constant(bn_b, tf.float32), [1, in_wid[blc], in_wid[blc], batch[blc]])
            conv = ten_a * conv + ten_b
            conv = tf.nn.relu(conv)
            if i == 0:
                print(blc," to ", blc+1," block. ")
                # print(conv)
            else:
                print(blc+1,"-th block, ", i,"-th layer")
                # print(conv)
            num += 1
            # print(i+1,"layer done\n")
        # print("after", blc_iter+1, "-th block\n", conv, "\n")

    ten_final = tf.reshape(tf.constant(np.loadtxt(in_dir+'final-fckernel.csv'), tf.float32), [1, 1, 64, 10])
    bias_final = tf.reshape(tf.constant(np.loadtxt(in_dir+'final-fcbias.csv'), tf.float32), [10])
    ten_pad = tf.reshape(tf.constant(pads[2], tf.float32), [1, in_wid[2], in_wid[2], batch[2]])
    # conv = conv * ten_pad
    # print(conv)
    conv = tf.reduce_mean(conv, [1,2], keepdims = True)
    conv = tf.nn.conv2d(conv, ten_final, strides = [1,1,1,1], padding = "SAME")
    conv = conv + bias_final

    conv = tf.squeeze(conv, axis=[1,2])
    # conv = tf.argmax(conv, 1)
    return conv

## load and save 
def load_save_data(num_samples):
    tf_labels = tf.constant(np.loadtxt('test_labels.csv'), tf.int64)
    tf_images = tf.reshape(tf.constant(np.loadtxt('test_images.csv'), tf.float32), [10000, 32, 32, 3])
    tf_labels_sm = tf_labels[:num_samples]
    tf_images_sm = tf_images[:num_samples,:,:,:]
    np.savetxt('test_labels_'+str(num_samples)+'.csv', tf_labels_sm, fmt='%d', delimiter=',')
    np.savetxt('test_images_'+str(num_samples)+'.csv', np.reshape(tf_images_sm, [-1]), fmt='%.18e', delimiter=',')

# compare enc result (after post process: reduce_mean) with plain result
def post_process(iter_num, ker_name, base_line):
    ## First load plain result
    # ker_name = 'ker7_'
    # base_line = False
    num_samples = 100
    pred = np.reshape(np.loadtxt('Resnet_plain_data/plain_prediction_'+ker_name+str(num_samples)+'.csv'), [num_samples, 10])    
    result_dir = 'Resnet_enc_result_'+ker_name
    if base_line:
        result_dir = result_dir + "BL"
    in_dir = 'weight_'+ker_name+'h5/'
    ten_final = tf.reshape(tf.constant(np.loadtxt(in_dir+'final-fckernel.csv'), tf.float32), [1, 1, 64, 10])
    bias_final = tf.reshape(tf.constant(np.loadtxt(in_dir+'final-fcbias.csv'), tf.float32), [10])

    acc = 0
    total = 0
    no_iters = []
    wrong_result = {}
    os_path = result_dir+'/class_result_'+ker_name
    if base_line:
        os_path = os_path+'BL_'
    for iter in range(iter_num):
        if os.path.exists(os_path+str(iter)+'.csv'):
            read = np.loadtxt(os_path+str(iter)+'.csv')
            total+=1
        else:
            no_iters.append(iter)
            continue
        if len(read) > 256:
            post_process = True
        else:
            post_process = False

        if post_process:
            conv = tf.reshape(tf.constant(read, tf.float32), [1, 16, 16, 256])
            conv = conv[:,:8,:8,:64]
            conv = tf.reduce_mean(conv, [1,2], keepdims = True)
            conv = tf.nn.conv2d(conv, ten_final, strides = [1,1,1,1], padding = "SAME")
            conv = conv + bias_final
            conv = tf.squeeze(conv, axis=[1,2])
            res_np = conv.numpy()
        else:
            res_np = read[:10] #np.reshape(read, [-1])[:10]
        print("enc: ", res_np, "argmax: ", np.argmax(res_np))
        print("plain: ", pred[iter], "argmax: ", np.argmax(pred[iter]))
        if (np.argmax(res_np) == np.argmax(pred[iter])):
            acc += 1
        else:
            wrong_result[str(iter)] = []
            wrong_result[str(iter)].insert(0, res_np)
            wrong_result[str(iter)].insert(1, pred[iter])

    print("precision: ", acc, "/", total)
    print("among ", iter_num, " samples.")
    print("missing: ", no_iters)
    print("\n wrong results: \n")
    for i, result in wrong_result.items():
        print(i, "-th iter.")
        print("enc: ", result[0], "argmax: ", np.argmax(result[0]))
        print("plain: ", result[1], "argmax: ", np.argmax(result[1]), "\n")

    # tf_images = tf.reshape(tf.constant(np.loadtxt('test_images_'+str(num_samples)+'.csv'), tf.float32), [num_samples, 32, 32, 3])
    # pred = plain_resnet(tf_images)
    # print("enc == plain?", tf.argmax(tf.squeeze(conv, axis=[1,2]),1) == tf.argmax(pred[iter],1))


# compare enc result (after post process: reduce_mean) with plain result
def post_process_Imgnet(iter_num, ker_name, base_line):
    ## First load plain result
    # ker_name = 'ker3'
    # base_line = False
    num_samples = 100
    true_pred = np.reshape(np.loadtxt('Imagenet/'+ker_name+'_true_label/'+ker_name+'_true_label0.csv'), [num_samples, 1000])    
    pred = np.reshape(np.loadtxt('Imagenet/'+ker_name+'_final_out/'+ker_name+'_final_out0.csv'), [num_samples, 1000])    
    result_dir = 'Imagenet/imgnet_class_result_'+ker_name+'_'
    if base_line:
        result_dir = result_dir + "BL"
    in_dir = 'weight_imgnet_'+ker_name+'_h5/'

    acc = 0
    total = 0
    no_iters = []
    wrong_result = {}
    os_path = result_dir+'/imgnet_enc_class_result_'+ker_name+'_'
    if base_line:
        os_path = os_path+'BL_'
    for iter in range(iter_num):
        if os.path.exists(os_path+str(iter)+'.csv'):
            read = np.loadtxt(os_path+str(iter)+'.csv')
            total+=1
        else:
            no_iters.append(iter)
            continue

        print("enc: ", read[:10], "argmax: ", np.argmax(read))
        print("plain: ", pred[iter,:][:10], "argmax: ", np.argmax(pred[iter,:]))
        if (np.argmax(read) == np.argmax(pred[iter,:])):
            acc += 1
        else:
            wrong_result[str(iter)] = []
            wrong_result[str(iter)].insert(0, read)
            wrong_result[str(iter)].insert(1, pred[iter,:])
            wrong_result[str(iter)].insert(2, true_pred[iter,:])

    print("precision: ", acc, "/", total)
    print("among ", iter_num, " samples.")
    print("missing: ", no_iters)
    print("\n wrong results: \n")
    for i, result in wrong_result.items():
        print(i, "-th iter.")
        print("enc: ", result[0][:10], "argmax: ", np.argmax(result[0]))
        print("plain: ", result[1][:10], "argmax: ", np.argmax(result[1]), "\n")
        print("true: argmax: ", np.argmax(result[2]), " \n" )

    # tf_images = tf.reshape(tf.constant(np.loadtxt('test_images_'+str(num_samples)+'.csv'), tf.float32), [num_samples, 32, 32, 3])
    # pred = plain_resnet(tf_images)
    # print("enc == plain?", tf.argmax(tf.squeeze(conv, axis=[1,2]),1) == tf.argmax(pred[iter],1))


def test_RMFC():
    batch = 64
    vec_size = 8*8*batch
    raw_input = [(i%53)/53.0 for i in range(vec_size)]
    raw_ker = [(i%13)/13.0 for i in range(batch*10)] #[1.0*i/640 for i in range(64*10)]
    bias = [10.0*i for i in range(10)]
    
    conv = tf.reshape(tf.constant(np.array(raw_input), tf.float32), [1, 8, 8, batch])
    print(conv)
    conv = tf.reduce_mean(conv, [1,2], keepdims = True)
    print("RM:", conv)
    ten_final = tf.reshape(tf.constant(np.array(raw_ker), tf.float32), [1,1,batch,10])
    print(ten_final)

    conv = tf.nn.conv2d(conv, ten_final, strides = [1,1,1,1], padding = "SAME") + tf.reshape(tf.constant(np.array(bias), tf.float32), [1, 1, 1, 10])
    print(conv)


def separate_data(num_outs):
    num_samples = 1000 # or 1000
    tf_labels = tf.constant(np.loadtxt('Resnet_plain_data/test_labels_'+str(num_samples)+'.csv'), tf.int64)
    tf_images = tf.reshape(tf.constant(np.loadtxt('Resnet_plain_data/test_images_'+str(num_samples)+'.csv'), tf.float32), [num_samples, 32, 32, 3])

    np.savetxt('test_data/test_labels.csv',tf_labels, fmt='%d', delimiter=',')
    for i in range(num_outs):
        np.savetxt('test_data/test_image_'+str(i)+'.csv',np.reshape(tf_images[i,:,:,:], [-1]), fmt='%.18e', delimiter=',')

def gen_plain_predictions():
    ker_name = 'ker7_'
    num_samples = 1000 # or 1000
    tf_labels = tf.constant(np.loadtxt('Resnet_plain_data/test_labels_'+str(num_samples)+'.csv'), tf.int64)
    tf_images = tf.reshape(tf.constant(np.loadtxt('Resnet_plain_data/test_images_'+str(num_samples)+'.csv'), tf.float32), [num_samples, 32, 32, 3])

    # np.savetxt('test_data/test_labels.csv',tf_labels, fmt='%d', delimiter=',')
    # for i in range(num_samples):
    #     np.savetxt('test_data/test_image_'+str(i)+'.csv',np.reshape(tf_images[i,:,:,:], [-1]), fmt='%.18e', delimiter=',')

    # predictions = plain_resnet(tf_images, ker_name)
    # tf_image = tf_images[0,:,:,:]
    predictions = plain_fast_resnet(tf_images, ker_name)
    # exit(1)
    # np.savetxt('Resnet_plain_data/plain_prediction_'+ker_name+str(num_samples)+'.csv',np.reshape(predictions, [-1]), fmt='%.18e', delimiter=',')
    print("num samples: ", len(tf_labels), "precision: ", tf.reduce_mean(tf.cast(tf.equal(tf.argmax(predictions, 1), tf_labels), 'float32')))

def imgnet_gen_ct_in_one(iter_st):
    ct_in = np.reshape(np.loadtxt('Imagenet/ker3_ct_in_2/ker3_ct_in'+str(iter_st)+'.csv'), [100, 14, 14, 256])
    j = 0
    for i in range(100*iter_st, 100*(iter_st+1)):
        np.savetxt("Imagenet/ker3_ct_in_one2/input_"+str(i)+".csv", np.reshape(ct_in[j,:,:,:], [-1]))
        j+=1

  
#### Main Start #### 


# conv_bnReLU_BL_bench(False, True, True)

# for i in range(5):
#     imgnet_gen_ct_in_one(i)


# # rename
# folder = "Resnet_enc_result_ker7_BL/"
# total_count = 0
# for count in range(100):
#     src = folder+"class_result_BL_ker7_"+str(count)+".csv"
#     dst = folder+"class_result_ker7_BL_"+str(count)+".csv"
#     os.rename(src, dst)
# exit(1)

# load_save_data(100)

# separate_data(300)
# trans_conv_bnReLU_BL_bench()
# conv_bnReLU_BL_bench(False)
# plain_resnet_bench()

# post_process_Imgnet(100, 'ker3', False)
# exit(1)

# plain_imagenet_bench()


# gen_plain_predictions()

# ## Imagnet test, take mid_input = np.loadtxt('Imagenet/ker3_ct_in/ker3_ct_in'+str(i)+'.csv')
# total = 0
# for i in range(10):
#     mid_input = np.loadtxt('Imagenet/ker3_ct_in/ker3_ct_in'+str(i)+'.csv')
#     # mid_input = np.loadtxt('Imagenet/ker3_mid_out/ker3_mid_out'+str(i)+'.csv')
#     mid_input = np.loadtxt('Imagenet/ker3_mid_out/ker3_mid_out'+str(i)+'.csv')
    # tf_mid_input = tf.reshape(tf.constant(mid_input, tf.float32), [100, 28, 28, 128])  
    # pred = plain_imagenet_after_mid(tf_mid_input, 'ker3_', False)
    # np.savetxt('Imagenet/ker3_final_out/ker3_final_out'+str(i)+'.csv', tf.reshape(pred, [-1]).numpy())
#     label = np.loadtxt('Imagenet/ker3_true_label/ker3_true_label'+str(i)+'.csv')
#     tf_label = tf.argmax(tf.reshape(tf.constant(label, tf.float32), [100, 1000]), 1)
#     correct = tf.reduce_sum(tf.cast(tf.equal(pred, tf_label), 'float32')).numpy()
#     total += correct
#     print("correct: ", correct, " among: 100")

# print("accuracy: ", total/1000.0)

#     mid_input = np.loadtxt('Imagenet/ker3_ct_in/ker3_ct_in'+str(0)+'.csv')
# mid_input = tf.reshape(tf.constant(np.loadtxt('Imagenet/ker3_ct_in/ker3_ct_in'+str(0)+'.csv'), tf.float32), [100, 28, 28, 128])
# label = tf.reshape(tf.constant(np.loadtxt('Imagenet/ker3_true_label/ker3_true_label'+str(0)+'.csv'), tf.float32), [100, 1000])

# for i in range(1, 10):
#     tmp_input = tf.reshape(tf.constant(np.loadtxt('Imagenet/ker3_ct_in/ker3_ct_in'+str(i)+'.csv'), tf.float32), [100, 28, 28, 128])
#     tmp_label = tf.reshape(tf.constant(np.loadtxt('Imagenet/ker3_true_label/ker3_true_label'+str(i)+'.csv'), tf.float32), [100, 1000])
#     mid_input = tf.concat([mid_input, tmp_input], 0)
#     label = tf.concat([label, tmp_label], 0)
# label = tf.argmax(tf.reshape(tf.constant(label, tf.float32), [1000, 1000]), 1)

# ## Imagnet test, take mid_input = np.loadtxt('Imagenet/ker3_ct_in/ker3_ct_in'+str(i)+'.csv')

# for i in range(10):
#     mid_input = np.loadtxt('Imagenet/ker3_ct_in/ker3_ct_in'+str(i)+'.csv')
#     tf_mid_input = tf.reshape(tf.constant(mid_input, tf.float32), [100, 28, 28, 128])  
#     plain_imagenet_after_mid(tf_mid_input, 'ker3_', False)
# exit(1)



# input_image = [random.uniform(0,1) for i in range(224*224*3)]
# input_image = tf.reshape(tf.constant(np.array(input_image), tf.float32), [1, 224, 224, 3])
# plain_imagenet(input_image, 'ker3_')

# post_process(100, 'ker7_', False)
# exit(1)
# num_samples = 1000
# pred = np.reshape(np.loadtxt('plain_prediction'+str(num_samples)+'.csv'), [num_samples, 10])    

# plain_imagenet_bench()
plain_imagenet_final_bench()
# test_RMFC()
# exit(1)

# trans = False
# strides = True
# relu = True
# conv_bnReLU_BL_bench(trans, strides, relu)
# exit(1)

# conv = np.reshape(np.loadtxt('class_result_'+str(5)+'.csv'), [256])
# print("enc: ", conv[:10], "argmax: ", np.argmax(conv[:10]))
# print("plain: ", pred[5], "argmax: ", np.argmax(pred[5]))

# gen_plain_predictions()