#### This file should be inside enc_test_imagenet

### evaluate plain python model
### save plain prediction
import tensorflow as tf
import numpy as np
import sys
import os
from tensorflow.keras.layers import Cropping2D
# from statistics import mean, stdev


## mid_input -> ct_input
## true_prediction -> argmax then save
def prep_imagenet(ker):
    num_sample = 2000
    weight_dir = 'weight_imgnet_ker'+str(ker)+'_h5/'
    mid_in_dir = '../ImageNet_ResNet_Tensorflow2.0/ker'+str(ker)+'_mid_out/ker'+str(ker)+'_mid_out0.csv'
    
    batch = [128, 256, 512]
    in_wid_list = {3: [28, 14, 7], 5: [28, 14, 6]}
    blc_list = {3: [4,4], 5: [2,2]}
    in_wid = in_wid_list[ker]
    blcs = blc_list[ker] 

    mid_input = tf.reshape(tf.constant(np.loadtxt(mid_in_dir), tf.float32), [num_sample, 28, 28, 128])
    num = 9
    bn_a = [[np.loadtxt(weight_dir+'w'+str(num)+'-a.csv') for i in range(in_wid[1])] for j in range(in_wid[1])]
    bn_b = [[np.loadtxt(weight_dir+'w'+str(num)+'-b.csv') for i in range(in_wid[1])] for j in range(in_wid[1])]
    ten_a = tf.reshape(tf.constant(bn_a, tf.float32), [1, in_wid[1], in_wid[1], batch[1]])
    ten_b = tf.reshape(tf.constant(bn_b, tf.float32), [1, in_wid[1], in_wid[1], batch[1]])
    mid_input = tf.nn.relu(ten_a * mid_input + ten_b)
    ten_k = tf.reshape(tf.constant(np.loadtxt(weight_dir+'w'+str(num)+'-conv.csv'), tf.float32), [ker, ker, batch[0], batch[1]])
    conv = tf.nn.conv2d(mid_input, ten_k, strides = [1,2,2,1], padding = "SAME")
    num+= 1
    bn_a = [[np.loadtxt(weight_dir+'w'+str(num)+'-a.csv') for i in range(in_wid[1])] for j in range(in_wid[1])]
    bn_b = [[np.loadtxt(weight_dir+'w'+str(num)+'-b.csv') for i in range(in_wid[1])] for j in range(in_wid[1])]
    ten_a = tf.reshape(tf.constant(bn_a, tf.float32), [1, in_wid[1], in_wid[1], batch[1]])
    ten_b = tf.reshape(tf.constant(bn_b, tf.float32), [1, in_wid[1], in_wid[1], batch[1]])
    conv = tf.nn.relu(ten_a * conv + ten_b)
    
    try:
        out_folder_dir = 'ker'+str(ker)+'_data'
        os.mkdir(out_folder_dir)        
    except OSError as error:
        print(error)

    np.savetxt(out_folder_dir+'/ct_in.csv', tf.reshape(conv, [-1]).numpy())

    tf_labels = tf.constant(np.loadtxt('../ImageNet_ResNet_Tensorflow2.0/ker'+str(ker)+'_true_label/ker'+str(ker)+'_true_label0.csv'), tf.int64, [num_sample, 1000])
    np.savetxt(out_folder_dir+'/true_label.csv', tf.reshape(tf.argmax(tf_labels, 1), [-1]).numpy())


# evaluate imagenet after having ct_in as input
def plain_imagenet_ct_in(ct_in, ker):
    weight_dir = 'weight_imgnet_ker'+str(ker)+'_h5/'
    
    batch = [128, 256, 512]
    in_wid_list = {3: [28, 14, 7], 5: [28, 14, 6]}
    blc_list = {3: [4,4], 5: [2,2]}
    in_wid = in_wid_list[ker]
    blcs = blc_list[ker] 

    num = 10
    conv = ct_in
    for i in range(blcs[0]-1):
        ten_k = tf.reshape(tf.constant(np.loadtxt(weight_dir+'w'+str(num)+'-conv.csv'), tf.float32), [ker, ker, batch[1], batch[1]])
        conv = tf.nn.conv2d(conv, ten_k, strides = [1,1,1,1], padding = "SAME")
        num+= 1
        bn_a = [[np.loadtxt(weight_dir+'w'+str(num)+'-a.csv') for i in range(in_wid[1])] for j in range(in_wid[1])]
        bn_b = [[np.loadtxt(weight_dir+'w'+str(num)+'-b.csv') for i in range(in_wid[1])] for j in range(in_wid[1])]
        ten_a = tf.reshape(tf.constant(bn_a, tf.float32), [1, in_wid[1], in_wid[1], batch[1]])
        ten_b = tf.reshape(tf.constant(bn_b, tf.float32), [1, in_wid[1], in_wid[1], batch[1]])
        conv = ten_a * conv + ten_b
        elements_gt = tf.math.greater(tf.reduce_max(tf.math.abs(conv), [1,2,3]),64.0)
        num_elements_gt = tf.math.reduce_sum(tf.cast(elements_gt, tf.int64))
        print("bigger than 64:", num_elements_gt)
        print(tf.reduce_max(tf.math.abs(conv), [0,1,2,3]))
        conv = tf.nn.relu(conv)

    for i in range(blcs[1]):
        if i == 0:
            ten_k = tf.reshape(tf.constant(np.loadtxt(weight_dir+'w'+str(num)+'-conv.csv'), tf.float32), [ker, ker, batch[1], batch[2]])
            conv = tf.nn.conv2d(conv, ten_k, strides = [1,2,2,1], padding = "SAME")
            if ker == 5:
                conv = Cropping2D(cropping=((0,1), (0,1)))(conv)
        else:
            ten_k = tf.reshape(tf.constant(np.loadtxt(weight_dir+'w'+str(num)+'-conv.csv'), tf.float32), [ker, ker, batch[2], batch[2]])
            conv = tf.nn.conv2d(conv, ten_k, strides = [1,1,1,1], padding = "SAME")
        num+= 1
        bn_a = [[np.loadtxt(weight_dir+'w'+str(num)+'-a.csv') for i in range(in_wid[2])] for j in range(in_wid[2])]
        bn_b = [[np.loadtxt(weight_dir+'w'+str(num)+'-b.csv') for i in range(in_wid[2])] for j in range(in_wid[2])]
        ten_a = tf.reshape(tf.constant(bn_a, tf.float32), [1, in_wid[2], in_wid[2], batch[2]])
        ten_b = tf.reshape(tf.constant(bn_b, tf.float32), [1, in_wid[2], in_wid[2], batch[2]])
        print(tf.reduce_max(tf.math.abs(ten_a*conv+ten_b), [0,1,2,3]))
        conv = tf.nn.relu(ten_a * conv + ten_b)

    ten_final = tf.reshape(tf.constant(np.loadtxt(weight_dir+'fc.csv'), tf.float32), [1, 1, batch[2], 1000])    
    conv = tf.reduce_mean(conv, [1,2], keepdims = True)
    conv = tf.nn.conv2d(conv, ten_final, strides = [1,1,1,1], padding = "SAME")

    conv = tf.squeeze(conv, axis=[1,2])
    return conv

# output num_samples of images, true_labels, plain_labels on which given resnet model has the full precision 
def gen_plain_predictions_imgnet(ker, target_prec):
    folder_dir = 'ker'+str(ker)+'_data/'

    total_samples = 2000
    num_samples_out = 1000 # [100, 1000, 10000]
    # target_prec = 0.679
    print("target precision: ", target_prec)

    tf_labels = tf.constant(np.loadtxt(folder_dir+'true_label.csv'), tf.int64)
    tf_ct_in = tf.constant(np.loadtxt(folder_dir+'ct_in.csv'), tf.float32, [total_samples, 14, 14, 256])
    print("load tensor done.")

    predictions = plain_imagenet_ct_in(tf_ct_in, ker)
    res_cpr = tf.equal(tf.argmax(predictions, 1), tf_labels).numpy()
    print("precision: ", np.sum(res_cpr)/(total_samples*1.0), ", total num: ", len(res_cpr))

    idx_list = np.array([False for i in range(total_samples)])
    idx = 0
    num_true = 0
    num_false = 0
    num_true_samples = int(target_prec*num_samples_out+0.5)
    while (num_true + num_false) < num_samples_out:
        if res_cpr[idx] and (num_true < num_true_samples):
            idx_list[idx] = True
            num_true += 1
        if (not res_cpr[idx]) and (num_false < (num_samples_out - num_true_samples)):
            idx_list[idx] = True
            num_false += 1
        idx += 1

    part_labels = tf.boolean_mask(tf_labels, idx_list)
    part_images = tf.boolean_mask(tf_ct_in, idx_list)
    part_predictions = plain_imagenet_ct_in(part_images, ker)

    print("num samples: ", len(part_labels), "precision: ", tf.reduce_mean(tf.cast(tf.equal(tf.argmax(part_predictions, 1), part_labels), 'float32')))
    

    if not os.path.exists(os.path.join(folder_dir, "test_labels_"+str(num_samples_out)+".csv")):
        np.savetxt(os.path.join(folder_dir, "test_labels_"+str(num_samples_out)+".csv"), np.reshape(part_labels, [-1]), fmt='%.18e', delimiter=',')        
    if not os.path.exists(os.path.join(folder_dir, "plain_prediction_"+str(num_samples_out)+".csv")):
        np.savetxt(os.path.join(folder_dir, "plain_prediction_"+str(num_samples_out)+".csv"), np.reshape(part_predictions, [-1]), fmt='%.18e', delimiter=',')        
    if not os.path.exists(os.path.join(folder_dir, "test_image_"+str(num_samples_out-1)+".csv")):
        for i in range(num_samples_out):
            np.savetxt(os.path.join(folder_dir, "test_image_"+str(i)+".csv"), np.reshape(part_images[i,:,:,:], [-1]), fmt='%.18e', delimiter=',')        


ker = int(sys.argv[1])
target_prec = float(sys.argv[2])
#prep_imagenet(ker)
gen_plain_predictions_imgnet(ker, target_prec)