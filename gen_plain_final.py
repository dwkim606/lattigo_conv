### evaluate plain python model
### save plain prediction
import tensorflow as tf
import numpy as np
import sys
import os
from tensorflow.keras.layers import Cropping2D
# from statistics import mean, stdev


def final_plain_resnet(input_image, crop, ker, depth, wid):
    if crop:
        in_dir = 'Resnet_weights/weights_crop_ker'+str(ker)+'_d'+str(depth)+'_wid'+str(wid)+'/'
    else:
        in_dir = 'Resnet_weights/weights_ker'+str(ker)+'_d'+str(depth)+'_wid'+str(wid)+'/'

    in_wid = [32, 16, 8]    
    init_batch = 16
    batch = [16*wid, 32*wid, 64*wid]
    
    blc_list = {20: [7,6,6], 14: [5,4,4], 8: [3,2,2]} # depends on depth
    pad_list = {3: [1,1,1], 5: [2,1,1], 7: [3,2,2]} # depends on ker
    ker_wid = ker
    ker_size = ker_wid**2
    blcs = blc_list[depth]
    pad_size = pad_list[ker]
    bn_pad_size = pad_size[0]
    if not crop:
        pad_size = [0, 0, 0]
        bn_pad_size = 0

    conv = input_image
    num = 0
    for blc in range(3):
        for i in range(blcs[blc]):
            if i == 0:
                if blc == 0:
                    ten_k = tf.reshape(tf.constant(np.loadtxt(in_dir+'w'+str(num)+'-conv.csv'), tf.float32), [ker_wid, ker_wid, 3, init_batch])
                    conv = Cropping2D(cropping=((0,pad_size[blc]), (0,pad_size[blc])))(conv)
                    conv = tf.nn.conv2d(conv, ten_k, strides = [1,1,1,1], padding = "SAME")
                else:
                    ten_k = tf.reshape(tf.constant(np.loadtxt(in_dir+'w'+str(num)+'-conv.csv'), tf.float32), [ker_wid, ker_wid, batch[blc-1], batch[blc]])
                    conv = tf.nn.conv2d(conv, ten_k, strides = [1,2,2,1], padding = "SAME")
                    conv = Cropping2D(cropping=((0,pad_size[blc]), (0,pad_size[blc])))(conv)
            elif (i == 1) and (blc == 0):
                ten_k = tf.reshape(tf.constant(np.loadtxt(in_dir+'w'+str(num)+'-conv.csv'), tf.float32), [ker_wid, ker_wid, init_batch, batch[blc]])
                conv = tf.nn.conv2d(conv, ten_k, strides = [1,1,1,1], padding = "SAME")
            else:
                ten_k = tf.reshape(tf.constant(np.loadtxt(in_dir+'w'+str(num)+'-conv.csv'), tf.float32), [ker_wid, ker_wid, batch[blc], batch[blc]])
                conv = tf.nn.conv2d(conv, ten_k, strides = [1,1,1,1], padding = "SAME")
            
            bn_a = [[np.loadtxt(in_dir+'w'+str(num)+'-a.csv') for i in range(in_wid[blc]-bn_pad_size)] for j in range(in_wid[blc]-bn_pad_size)]
            bn_b = [[np.loadtxt(in_dir+'w'+str(num)+'-b.csv') for i in range(in_wid[blc]-bn_pad_size)] for j in range(in_wid[blc]-bn_pad_size)]
            if (blc == 0) and (i == 0):
                ten_a = tf.reshape(tf.constant(bn_a, tf.float32), [1, in_wid[blc]-bn_pad_size, in_wid[blc]-bn_pad_size, init_batch])
                ten_b = tf.reshape(tf.constant(bn_b, tf.float32), [1, in_wid[blc]-bn_pad_size, in_wid[blc]-bn_pad_size, init_batch])    
            else:
                ten_a = tf.reshape(tf.constant(bn_a, tf.float32), [1, in_wid[blc]-bn_pad_size, in_wid[blc]-bn_pad_size, batch[blc]])
                ten_b = tf.reshape(tf.constant(bn_b, tf.float32), [1, in_wid[blc]-bn_pad_size, in_wid[blc]-bn_pad_size, batch[blc]])
            conv = ten_a * conv + ten_b
            # elements_gt = tf.math.greater(tf.reduce_max(tf.math.abs(conv), [1,2,3]),32.0)
            # num_elements_gt = tf.math.reduce_sum(tf.cast(elements_gt, tf.int32))
            # print("bigger than 32:", num_elements_gt)
            print("layer:", num, "max:", tf.reduce_max(conv, [0,1,2,3]))
            conv = tf.nn.relu(conv)
            # print(i, blc, conv)
            # if i == 0:
            #     print(blc," to ", blc+1," block. ")
            #     # print(conv)
            # else:
            #     print(blc+1,"-th block, ", i,"-th layer")
            #     # print(conv)
            num += 1
            # print(i+1,"layer done\n")
        # print("after", blc_iter+1, "-th block\n", conv, "\n")

    # conv = conv * ten_pad # zeroizing the relus(ten_b) part!!
    ten_final = tf.reshape(tf.constant(np.loadtxt(in_dir+'final-fckernel.csv'), tf.float32), [1, 1, batch[2], 10])
    bias_final = tf.reshape(tf.constant(np.loadtxt(in_dir+'final-fcbias.csv'), tf.float32), [10])
    conv = tf.reduce_mean(conv, [1,2], keepdims = True)
    conv = tf.nn.conv2d(conv, ten_final, strides = [1,1,1,1], padding = "SAME")
    conv = conv + bias_final

    conv = tf.squeeze(conv, axis=[1,2])
    return conv

# output num_samples of images, true_labels, plain_labels on which given resnet model has the full precision 
# generate folder named "ker3_d8_wid1" inside Resnet_plain_data, then test_labels_100, plain_prediction_100, test_image_0.csv, ..., .
def final_gen_plain_predictions(crop, ker, depth, wid):
    num_samples = 1000 # [100, 1000, 10000]
    tf_labels = tf.constant(np.loadtxt('Resnet_plain_data/test_labels.csv'), tf.int64)
    tf_images = tf.constant(np.loadtxt('Resnet_plain_data/test_images.csv'), tf.float32, [10000, 32, 32, 3])
    # else:
    #     tf_labels = tf.constant(np.loadtxt('Resnet_plain_data/test_labels_'+str(num_samples)+'.csv'), tf.int64)
    #     tf_images = tf.constant(np.loadtxt('Resnet_plain_data/test_images_'+str(num_samples)+'.csv'), tf.float32, [num_samples, 32, 32, 3])

    predictions = final_plain_resnet(tf_images, crop, ker, depth, wid)
    full_prec = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(predictions, 1), tf_labels), 'float32')).numpy()
    print("full precision: ", full_prec)
    if num_samples == 1000:
        full_prec_int = int(full_prec * num_samples + 0.5)
    elif num_samples == 100:
        full_prec_int = int(full_prec * num_samples + 0.5)
    else:
        print("wrong num_samples (only 100 or 1000)")
        exit(1)

    res_cpr = tf.equal(tf.argmax(predictions, 1), tf_labels).numpy()
    idx_list = np.array([False for i in range(10000)])
    idx = 0
    num_true = 0
    num_false = 0
    print("full prec int: ", full_prec_int)
    while (num_true + num_false) < num_samples:
        if res_cpr[idx] and (num_true < full_prec_int):
            idx_list[idx] = True
            num_true += 1
        if (not res_cpr[idx]) and (num_false < (num_samples - full_prec_int)):
            idx_list[idx] = True
            num_false += 1
        idx += 1

    part_labels = tf.boolean_mask(tf_labels, idx_list)
    part_images = tf.boolean_mask(tf_images, idx_list)
    part_predictions = final_plain_resnet(part_images, crop, ker, depth, wid)

    print("num samples: ", len(part_labels), "precision: ", tf.reduce_mean(tf.cast(tf.equal(tf.argmax(part_predictions, 1), part_labels), 'float32')))

    out_folder = 'ker'+str(ker)+'_d'+str(depth)+'_wid'+str(wid)
    if crop:
        out_folder = 'crop_' + out_folder
    try:
        out_folder_dir = os.path.join('Resnet_plain_data', out_folder)
        os.mkdir(out_folder_dir)        
    except OSError as error:
        print(error)

    if not os.path.exists(os.path.join(out_folder_dir, "test_labels_"+str(num_samples)+".csv")):
        np.savetxt(os.path.join(out_folder_dir, "test_labels_"+str(num_samples)+".csv"), np.reshape(part_labels, [-1]), fmt='%.18e', delimiter=',')        
    if not os.path.exists(os.path.join(out_folder_dir, "plain_prediction_"+str(num_samples)+".csv")):
        np.savetxt(os.path.join(out_folder_dir, "plain_prediction_"+str(num_samples)+".csv"), np.reshape(part_predictions, [-1]), fmt='%.18e', delimiter=',')        
    if not os.path.exists(os.path.join(out_folder_dir, "test_image_"+str(num_samples-1)+".csv")):
        for i in range(num_samples):
            np.savetxt(os.path.join(out_folder_dir, "test_image_"+str(i)+".csv"), np.reshape(part_images[i,:,:,:], [-1]), fmt='%.18e', delimiter=',')        
    

crop = True
ker = int(sys.argv[1])
depth = int(sys.argv[2])
wide = int(sys.argv[3])

final_gen_plain_predictions(crop, ker, depth, wide)
