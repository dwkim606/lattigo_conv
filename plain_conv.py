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

def prt_list(input, start, width, showAll):
    if showAll:
        for i in range(len(input) // width):
            if (i%(width) == 0):
                print()
            print(input[start + i * width: start + i * width + width])
    else:
        for i in range(width):
            print(input[start + i * width: start + i * width + width])

def plain_imagenet_bench():
    input_width = 16
    num_bl1 = 2
    num_bl2 = 2
    batch = 16
    vec_size = batch*input_width**2
    ker_width = 3
    bn_a = [0.3, 0.1]

    batch2 = batch*2
    ker_size = ker_width**2

    print("input width:", input_width)
    print("batch:", batch)
    print("ker width:", ker_width)
    #data = np.loadtxt("./weights/batch0.txt")

    ## Correctness Check: Compare with TF NN CONV2D
    raw_input = [(1.0 * i)/vec_size for i in range(vec_size)]
    ker =  [(0.1 * i)/(batch * batch * ker_size) for i in range(batch * batch * ker_size)] #[0.1 * i / (batch * batch * filter_size) for i in range(batch * batch * filter_size)]
    ker12 = [(0.1 * i)/(batch * batch2 * ker_size) for i in range(batch * batch2 * ker_size)] #[0.1 * i / (batch * batch * filter_size) for i in range(batch * batch * filter_size)]
    ker2 = [(0.1 * i)/(batch2 * batch2 * ker_size) for i in range(batch2 * batch2 * ker_size)] #[0.1 * i / (batch * batch * filter_size) for i in range(batch * batch * filter_size)]

    ten_x = tf.reshape(tf.constant(np.array(raw_input), tf.float32), [1, input_width, input_width, batch])
    ten_k = tf.reshape(tf.constant(np.array(ker), tf.float32), [ker_width, ker_width, batch, batch])
    ten_k12 = tf.reshape(tf.constant(np.array(ker12), tf.float32), [ker_width, ker_width, batch, batch2])
    ten_k2 = tf.reshape(tf.constant(np.array(ker2), tf.float32), [ker_width, ker_width, batch2, batch2])

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


def plain_resnet_bench():
    input_width = 32
    batch = 4
    vec_size = 3*input_width**2
    ker_width = 5
    bn_a = [0.2, 0.2, 0.1]

    batch2 = batch*2
    batch3 = batch2*2
    ker_size = ker_width**2

    print("input width:", input_width)
    print("batch:", batch)
    print("ker width:", ker_width)
    #data = np.loadtxt("./weights/batch0.txt")

    ## Correctness Check: Compare with TF NN CONV2D
    raw_input = [(5.0 * i)/vec_size for i in range(vec_size)]
    ker0 = [(0.25 * i)/(3 * batch * ker_size) for i in range(3 * batch * ker_size)] #[0.1 * i / (batch * batch * filter_size) for i in range(batch * batch * filter_size)]
    ker =  [(0.25 * i)/(batch * batch * ker_size) for i in range(batch * batch * ker_size)] #[0.1 * i / (batch * batch * filter_size) for i in range(batch * batch * filter_size)]
    ker12 = [(0.25 * i)/(batch * batch2 * ker_size) for i in range(batch * batch2 * ker_size)] #[0.1 * i / (batch * batch * filter_size) for i in range(batch * batch * filter_size)]
    ker2 = [(0.25 * i)/(batch2 * batch2 * ker_size) for i in range(batch2 * batch2 * ker_size)] #[0.1 * i / (batch * batch * filter_size) for i in range(batch * batch * filter_size)]
    ker23 = [(0.25 * i)/(batch2 * batch3 * ker_size) for i in range(batch2 * batch3 * ker_size)]  # (0.5 * i)/(batch2 * batch3 * ker_size) #[0.1 * i / (batch * batch * filter_size) for i in range(batch * batch * filter_size)]
    ker3 = [(0.25 * i)/(batch3 * batch3 * ker_size) for i in range(batch3 * batch3 * ker_size)] #[0.1 * i / (batch * batch * filter_size) for i in range(batch * batch * filter_size)]

    ten_x = tf.reshape(tf.constant(np.array(raw_input), tf.float32), [1, input_width, input_width, 3])
    # print(ten_x)
    ten_k0 = tf.reshape(tf.constant(np.array(ker0), tf.float32), [ker_width, ker_width, 3, batch])
    ten_k = tf.reshape(tf.constant(np.array(ker), tf.float32), [ker_width, ker_width, batch, batch])
    ten_k12 = tf.reshape(tf.constant(np.array(ker12), tf.float32), [ker_width, ker_width, batch, batch2])
    ten_k2 = tf.reshape(tf.constant(np.array(ker2), tf.float32), [ker_width, ker_width, batch2, batch2])
    ten_k23 = tf.reshape(tf.constant(np.array(ker23), tf.float32), [ker_width, ker_width, batch2, batch3])
    ten_k3 = tf.reshape(tf.constant(np.array(ker3), tf.float32), [ker_width, ker_width, batch3, batch3])
    # print(ten_k)

    # conv = tf.nn.conv2d_transpose(ten_x, ten_k1, output_shape=(1, input_width[1], input_width[1], batch[1]), strides=[1, 2, 2, 1])

    num_bl1 = 5
    num_bl2 = 3
    num_bl3 = 3

    conv = ten_x
    print("Input: ", conv)
    for i in range(num_bl1):
        if i == 0:
            conv = tf.nn.conv2d(conv, ten_k0, strides = [1,1,1,1], padding = "SAME")*bn_a[0]    
        else:
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

def conv_bnReLU_BL_bench(trans, strides, relu):
    batch = 8
    input_width = 4
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

def plain_resnet(input_image):
    in_dir = 'weight_h5/'
    in_wid = [32, 16, 8]
    batch = [16, 32, 64]
    ker_wid = 3
    ker_size = ker_wid**2
    
    # load weights
    # blocks = [1, 2, 3]
    # units = range(2)
    # kinds = ['a', 'b', 'conv']

    # raw_input = [0.1*i/3000.0 for i in range(32*32*3)]
    # conv = tf.reshape(tf.constant(np.array(raw_input), tf.float32), [1, in_wid[0], in_wid[0], 3])
    conv = input_image
    blcs = [7,6,6] #ker7 [3,2,2] #ker5 [5,4,4] #ker3 [7,6,6]

    # print("Input: ", conv)

    num = 0
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
            else:
                print(blc+1,"-th block, ", i,"-th layer")
            num += 1
            # print(i+1,"layer done\n")
        # print("after", blc_iter+1, "-th block\n", conv, "\n")

    ten_final = tf.reshape(tf.constant(np.loadtxt(in_dir+'final-fckernel.csv'), tf.float32), [1, 1, 64, 10])
    bias_final = tf.reshape(tf.constant(np.loadtxt(in_dir+'final-fcbias.csv'), tf.float32), [10])
    conv = tf.reduce_mean(conv, [1,2], keepdims = True)
    conv = tf.nn.conv2d(conv, ten_final, strides = [1,1,1,1], padding = "SAME")
    conv = conv + bias_final

    conv = tf.squeeze(conv, axis=[1,2])
    conv = tf.argmax(conv, 1)
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
def post_process(iter_num):
    ## First load plain result
    num_samples = 1000
    pred = np.reshape(np.loadtxt('Resnet_plain_data/plain_prediction'+str(num_samples)+'.csv'), [num_samples, 10])    
    result_dir = 'Resnet_test_result_enc_BL/'
    in_dir = 'weight_h5/'
    ten_final = tf.reshape(tf.constant(np.loadtxt(in_dir+'final-fckernel.csv'), tf.float32), [1, 1, 64, 10])
    bias_final = tf.reshape(tf.constant(np.loadtxt(in_dir+'final-fcbias.csv'), tf.float32), [10])

    acc = 0
    total = 0
    no_iters = []
    wrong_result = {}
    for iter in range(iter_num):
        if os.path.exists(result_dir+'class_result_BL_'+str(iter)+'.csv'):
            read = np.loadtxt(result_dir+'class_result_BL_'+str(iter)+'.csv')
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

def test_RMFC():
    vec_size = 8*8*64
    raw_input = [(i%27)*0.1 for i in range(vec_size)]
    raw_ker = [(i%27)*0.1 for i in range(64*10)] #[1.0*i/640 for i in range(64*10)]
    bias = [1*i for i in range(10)]
    
    conv = tf.reshape(tf.constant(np.array(raw_input), tf.float32), [1, 8, 8, 64])
    print(conv)
    conv = tf.reduce_mean(conv, [1,2], keepdims = True)
    print("RM:", conv)
    ten_final = tf.reshape(tf.constant(np.array(raw_ker), tf.float32), [1,1,64,10])
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

#### Main Start #### 

# load_save_data(100)

# separate_data(300)
# trans_conv_bnReLU_BL_bench()
# conv_bnReLU_BL_bench(False)
# plain_resnet_bench()
# exit(1)
# post_process(100)
# exit(1)
# num_samples = 1000
# pred = np.reshape(np.loadtxt('plain_prediction'+str(num_samples)+'.csv'), [num_samples, 10])    

# plain_imagenet_bench()
# test_RMFC()

# trans = False
# strides = True
# relu = False
# conv_bnReLU_BL_bench(trans, strides, relu)

# conv = np.reshape(np.loadtxt('class_result_'+str(5)+'.csv'), [256])
# print("enc: ", conv[:10], "argmax: ", np.argmax(conv[:10]))
# print("plain: ", pred[5], "argmax: ", np.argmax(pred[5]))

num_samples = 1000 # or 1000
tf_labels = tf.constant(np.loadtxt('Resnet_plain_data/test_labels_'+str(num_samples)+'.csv'), tf.int64)
tf_images = tf.reshape(tf.constant(np.loadtxt('Resnet_plain_data/test_images_'+str(num_samples)+'.csv'), tf.float32), [num_samples, 32, 32, 3])

# np.savetxt('test_data/test_labels.csv',tf_labels, fmt='%d', delimiter=',')
# for i in range(num_samples):
#     np.savetxt('test_data/test_image_'+str(i)+'.csv',np.reshape(tf_images[i,:,:,:], [-1]), fmt='%.18e', delimiter=',')
    
# tf_image = tf.reshape(tf_images[0, :, :, :], [1,32,32,3])
# print(tf_image)
predictions = plain_resnet(tf_images)
np.savetxt('plain_prediction_'+str(num_samples)+'.csv',np.reshape(predictions, [-1]), fmt='%.18e', delimiter=',')
print("num samples: ", len(tf_labels), "precision: ", tf.reduce_mean(tf.cast(tf.equal(predictions, tf_labels), 'float32')))

