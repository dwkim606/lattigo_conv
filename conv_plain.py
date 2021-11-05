# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import numpy as np
import time
import sys
import math
import cmath
import random
import tensorflow as tf



# from (1) format to (H,W,batch)
# one ciphertext
def reshape_input(input, width, batch):
    output = [0] * vec_size
    l = 0
    for i in range(width):
        for j in range(width):
            for k in range(batch):
                output[j + i * width + k * width * width] = input[l]
                l += 1

    return output

#from (1) format to (H,W,in,out)
def reshape_filter(input, num_input, num_output):
    return [[[[input[i + j * num_input + k * num_input * num_output + l * num_input * num_output * ker_width] for i in range(num_input)] for j in range(num_output)] for k in range(ker_width)] for l in range(ker_width)]


def prt_list(input, start, width, showAll):
    if showAll:
        for i in range(len(input) // width):
            if (i%(width) == 0):
                print()
            print(input[start + i * width: start + i * width + width])
    else:
        for i in range(width):
            print(input[start + i * width: start + i * width + width])


input_width = 4
batch = 4
vec_size = batch*input_width**2
ker_width = 3
bn_a = 0.1

batch_out = batch
ker_size = ker_width**2

print("input width:", input_width)
print("batch:", batch)
print("ker width:", ker_width)
#data = np.loadtxt("./weights/batch0.txt")

#print(data)

#input_width = [4, 8, 16, 32, 64]
#input_size = [16, 64, 256, 1024, 4096]
#cnum = [1, 2, 4, 8, 3]
#batch = [1024, 256, 64, 16, 1]

## Correctness Check: Compare with TF NN CONV2D
raw_input = [(1.0 * i)/vec_size for i in range(vec_size)]
ker =  [(1.0 * i)/(batch * batch_out * ker_size) for i in range(batch * batch_out * ker_size)] #[0.1 * i / (batch * batch * filter_size) for i in range(batch * batch * filter_size)]

ten_x = tf.reshape(tf.constant(np.array(raw_input), tf.float32), [1, input_width, input_width, batch])
# print(ten_x)
ten_k = tf.reshape(tf.constant(np.array(ker), tf.float32), [ker_width, ker_width, batch, batch_out])
# print(ten_k)

# print(ten_k[:,:,:,0])
# print(ten_k[:,:,:,1])
# print(ten_k[:,:,:,2])

num_bl1 = 1

conv = ten_x
for i in range(num_bl1):
    conv = tf.nn.conv2d(conv, ten_k, strides = [1,1,1,1], padding = "SAME")*bn_a
    conv = tf.nn.relu(conv)
    print(i+1,"layer done\n")
print("after 1layer", conv)

# conv = tf.nn.conv2d_transpose(ten_x, ten_k1, output_shape=(1, input_width[1], input_width[1], batch[1]), strides=[1, 2, 2, 1])


conv = tf.nn.conv2d(conv, ten_k, strides = [1,2,2,1], padding = "SAME")*bn_a
conv = tf.nn.relu(conv)
print("after 2layer", conv)