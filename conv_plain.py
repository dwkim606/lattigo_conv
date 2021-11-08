# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import numpy as np
import time
import sys
import math
import cmath
import random
import tensorflow as tf

def prt_list(input, start, width, showAll):
    if showAll:
        for i in range(len(input) // width):
            if (i%(width) == 0):
                print()
            print(input[start + i * width: start + i * width + width])
    else:
        for i in range(width):
            print(input[start + i * width: start + i * width + width])

def plain_conv_resnet():
    input_width = 32
    batch = 4
    vec_size = batch*input_width**2
    ker_width = 3
    bn_a = [0.02, 0.02, 0.02]

    batch2 = batch*2
    batch3 = batch2*2
    ker_size = ker_width**2

    print("input width:", input_width)
    print("batch:", batch)
    print("ker width:", ker_width)
    #data = np.loadtxt("./weights/batch0.txt")

    ## Correctness Check: Compare with TF NN CONV2D
    raw_input = [(1.0 * i)/vec_size for i in range(vec_size)]
    ker =  [(0.5 * i)/(batch * batch * ker_size) for i in range(batch * batch * ker_size)] #[0.1 * i / (batch * batch * filter_size) for i in range(batch * batch * filter_size)]
    ker12 = [(0.5 * i)/(batch * batch2 * ker_size) for i in range(batch * batch2 * ker_size)] #[0.1 * i / (batch * batch * filter_size) for i in range(batch * batch * filter_size)]
    ker2 = [(0.5 * i)/(batch2 * batch2 * ker_size) for i in range(batch2 * batch2 * ker_size)] #[0.1 * i / (batch * batch * filter_size) for i in range(batch * batch * filter_size)]
    ker23 = [(0.5*i)/(batch2 * batch3 * ker_size) for i in range(batch2 * batch3 * ker_size)] #[0.1 * i / (batch * batch * filter_size) for i in range(batch * batch * filter_size)]
    ker3 = [(0.5 * i)/(batch3 * batch3 * ker_size) for i in range(batch3 * batch3 * ker_size)] #[0.1 * i / (batch * batch * filter_size) for i in range(batch * batch * filter_size)]

    ten_x = tf.reshape(tf.constant(np.array(raw_input), tf.float32), [1, input_width, input_width, batch])
    # print(ten_x)
    ten_k = tf.reshape(tf.constant(np.array(ker), tf.float32), [ker_width, ker_width, batch, batch])
    ten_k12 = tf.reshape(tf.constant(np.array(ker12), tf.float32), [ker_width, ker_width, batch, batch2])
    ten_k2 = tf.reshape(tf.constant(np.array(ker2), tf.float32), [ker_width, ker_width, batch2, batch2])
    ten_k23 = tf.reshape(tf.constant(np.array(ker23), tf.float32), [ker_width, ker_width, batch2, batch3])
    ten_k3 = tf.reshape(tf.constant(np.array(ker3), tf.float32), [ker_width, ker_width, batch3, batch3])
    # print(ten_k)

    # conv = tf.nn.conv2d_transpose(ten_x, ten_k1, output_shape=(1, input_width[1], input_width[1], batch[1]), strides=[1, 2, 2, 1])

    num_bl1 = 1
    num_bl2 = 1
    num_bl3 = 1

    conv = ten_x
    print("Input: ", conv)
    for i in range(num_bl1):
        conv = tf.nn.conv2d(conv, ten_k, strides = [1,1,1,1], padding = "SAME")*bn_a[0]
        conv = tf.nn.relu(conv)
        print(i+1,"layer done\n")
    print("after 1st block\n", conv, "\n")

    conv = tf.nn.conv2d(conv, ten_k12, strides = [1,2,2,1], padding = "SAME")*bn_a[1]
    conv = tf.nn.relu(conv)
    print("after 1st to 2nd block\n", conv, "\n")

    for i in range(num_bl2):
        conv = tf.nn.conv2d(conv, ten_k2, strides = [1,1,1,1], padding = "SAME")*bn_a[1]
        conv = tf.nn.relu(conv)
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
        print(i+1,"layer done\n")
    print("after 3rd block\n", conv, "\n")


#### Main Start #### 

input_width = 16
batch = 128
vec_size = batch*input_width**2
ker_width = 3
bn_a = 0.001

ker_size = ker_width**2 

print("input width:", input_width)
print("batch:", batch)
print("ker width:", ker_width)
#data = np.loadtxt("./weights/batch0.txt")

## Correctness Check: Compare with TF NN CONV2D
raw_input = [1.0*i/vec_size for i in range(vec_size)]
ker =  [1.0*i/(batch * batch * ker_size) for i in range(batch * batch * ker_size)] #[0.1 * i / (batch * batch * filter_size) for i in range(batch * batch * filter_size)]

ten_x = tf.reshape(tf.constant(np.array(raw_input), tf.float32), [1, input_width, input_width, batch])
print("input: \n", ten_x)
ten_k = tf.reshape(tf.constant(np.array(ker), tf.float32), [ker_width, ker_width, batch, batch])
# print("kernel: \n", ten_k)

# print("ker0: ", ten_k[:,:,0,0])
conv = tf.nn.conv2d(ten_x, ten_k, strides = [1,1,1,1], padding = "SAME")*bn_a
conv = tf.nn.relu(conv)
print("result: \n", conv, "\n")