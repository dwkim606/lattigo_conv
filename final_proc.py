### evaluate plain python model
### save plain prediction
### Compare with encrypted prediction
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.layers import Cropping2D
from statistics import mean, stdev

## precision
prec = 2

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

def final_gen_plain_predictions(crop, ker, depth, wid):
    num_samples = 1000 # or 1000
    if num_samples == 10000:
        tf_labels = tf.constant(np.loadtxt('Resnet_plain_data/test_labels.csv'), tf.int64)
        tf_images = tf.reshape(tf.constant(np.loadtxt('Resnet_plain_data/test_images.csv'), tf.float32), [num_samples, 32, 32, 3])
    else:
        tf_labels = tf.constant(np.loadtxt('Resnet_plain_data/test_labels_'+str(num_samples)+'.csv'), tf.int64)
        tf_images = tf.reshape(tf.constant(np.loadtxt('Resnet_plain_data/test_images_'+str(num_samples)+'.csv'), tf.float32), [num_samples, 32, 32, 3])

    # tf_images = tf.reshape(tf_images[0,:,:,:], [1,32,32,3])
    # tf_labels = tf_labels[:200]

    predictions = final_plain_resnet(tf_images, crop, ker, depth, wid)
    # predictions = plain_fast_resnet(tf_images, ker_name)

    if crop:
        out_filename = 'Resnet_plain_data/plain_prediction_crop_ker'+str(ker)+'_d'+str(depth)+'_wid'+str(wid)+'_'+str(num_samples)+'.csv'
    else:
        out_filename = 'Resnet_plain_data/plain_prediction_ker'+str(ker)+'_d'+str(depth)+'_wid'+str(wid)+'_'+str(num_samples)+'.csv'

    if num_samples != 10000:
        if not os.path.exists(out_filename):
            np.savetxt(out_filename,np.reshape(predictions, [-1]), fmt='%.18e', delimiter=',')        
    
    print("num samples: ", len(tf_labels), "precision: ", tf.reduce_mean(tf.cast(tf.equal(tf.argmax(predictions, 1), tf_labels), 'float32')))


# for resnet crop, compare plain with enc
def final_post_process(iter_num, crop, ker, depth, wid):    
    max_num_samples = 1000

    if crop:
        plain_pred_file = 'Resnet_plain_data/plain_prediction_crop_ker'+str(ker)+'_d'+str(depth)+'_wid'+str(wid)+'_'+str(max_num_samples)+'.csv'
        enc_result_dir = 'result_crop_ker'+str(ker)+'_d'+str(depth)+'_wid'+str(wid)+'/'
    else:
        plain_pred_file = 'Resnet_plain_data/plain_prediction_ker'+str(ker)+'_d'+str(depth)+'_wid'+str(wid)+'_'+str(max_num_samples)+'.csv'
        enc_result_dir = 'result_ker'+str(ker)+'_d'+str(depth)+'_wid'+str(wid)+'/'

    plain_pred = np.reshape(np.loadtxt(plain_pred_file), [max_num_samples, 10])    
    true_pred = np.reshape(np.loadtxt('Resnet_plain_data/test_labels_'+str(max_num_samples)+'.csv'), [max_num_samples])    

    acc = 0
    true_acc = 0
    pl_true_acc = 0
    total = 0
    no_iters = []
    wrong_result = {}
    os_path = enc_result_dir+'/class_result_ker'+str(ker)+'_'

    for iter in range(iter_num):
        if os.path.exists(os_path+str(iter)+'.csv'):
            read = np.loadtxt(os_path+str(iter)+'.csv')
            total+=1
        else:
            no_iters.append(iter)
            continue

        res_np = read[:10] #np.reshape(read, [-1])[:10]
        print("enc: ", res_np, "argmax: ", np.argmax(res_np))
        print("plain: ", plain_pred[iter], "argmax: ", np.argmax(plain_pred[iter]))
        if (np.argmax(res_np) == np.argmax(plain_pred[iter])):
            acc += 1
        else:
            wrong_result[str(iter)] = []
            wrong_result[str(iter)].insert(0, res_np)
            wrong_result[str(iter)].insert(1, plain_pred[iter])
            wrong_result[str(iter)].insert(2, true_pred[iter])
        if (np.argmax(res_np) == true_pred[iter]):
            true_acc += 1
        if (np.argmax(plain_pred[iter]) == true_pred[iter]):
            pl_true_acc += 1

    print("Plain precision: ", pl_true_acc, "/", total)
    print("Enc precision: ", true_acc, "/", total)
    print("plain vs enc accordance: ", acc, "/", total)
    print("among ", iter_num, " samples.")
    print("missing: ", no_iters)
    print("\n wrong results: \n")
    for i, result in wrong_result.items():
        print(i, "-th iter.")
        print("enc: ", result[0], "argmax: ", np.argmax(result[0]))
        print("plain: ", result[1], "argmax: ", np.argmax(result[1]), "\n")
        print("true: ", result[2], " \n" )

    # tf_images = tf.reshape(tf.constant(np.loadtxt('test_images_'+str(num_samples)+'.csv'), tf.float32), [num_samples, 32, 32, 3])
    # pred = plain_resnet(tf_images)
    # print("enc == plain?", tf.argmax(tf.squeeze(conv, axis=[1,2]),1) == tf.argmax(pred[iter],1))


def read_out_analysis_time(prefix, os_path):
    # prefix = "After CtS    :"
    # prefix = "After Sine   :"
    # prefix = "After StC    : "
    # prefix = "Eval: Relu Done in"

    # prefix = "Total done in "
    # prefix = "Done in "
    # prefix = "Conv (with BN) Done in" 
    # prefix = "(until CtoS):"
    # prefix = "Eval: Eval: ReLU Done in"
    # prefix = "Boot (StoC) Done in "
    # prefix = "AVG Prec : ("

    list_results = []
    if os.path.exists(os_path):
        with open(os_path, 'r') as read_obj:
            for line in read_obj:
                if "iter" in line:
                    new_iter = True
                    count = 0           ## count the number of appearence in each iter
                if prefix in line:
                    count += 1
                    if prefix == "(until CtoS):":
                        time_str = next(read_obj,'').strip("Done in")
                    else:
                        time_str = line.strip(prefix)
                    
                    if new_iter:
                        list_results.append(get_seconds(time_str))
                        new_iter = False
                    else:
                        list_results[-1] += get_seconds(time_str)
    else:
        print("No file exists")
        exit(1)
    
    return(count, list_results)

# read precision
def read_out_analysis_prec():
    os_path = 'Resnet_enc_result_ker3_'+'/total200.out'
    # os_path = 'test_convRelu_ker5.out'

    # prefix = "Evaluation total done in "
    # prefix = "After CtS    :"
    # prefix = "After Sine   :"
    # prefix = "After StC    : "
    # prefix = "Eval: Relu Done in"

    # prefix = "Total done in "
    # prefix = "Done in "
    # prefix = "Conv (with BN) Done in" 
    # prefix = "(until CtoS):"
    # prefix = "Eval: Eval: ReLU Done in"
    # prefix = "Boot (StoC) Done in "
    # prefix = "AVG Prec : ("

    line_number = 0
    list_results = [] 
    base_line = False
    if os.path.exists(os_path):
        with open(os_path, 'r') as read_obj:
            for line in read_obj:
                if "BL" in line:
                    base_line = True
                if not base_line:
                    if prefix in line:
                        # try:
                        #     time_str = line.strip(prefix).rstrip()
                        #     if line_number%6 == 0:
                        #         list_results.append(get_seconds(time_str))
                        #     line_number += 1
                        # except:
                        #     continue
                        line_number += 1
                        time_str, _ = line.strip(prefix).split(',') # for prec
                        if line_number%7 == 0:
                            list_results.append(float(time_str))      # for prec
                        
    else:
        print("No file exists")
        exit(1)
    
    return(list_results)


def get_seconds(time_str):
    try:
        ms, _ = time_str.split('ms')
        try:
            s, ms = ms.split('s')
            try: 
                m, s = s.split('m')
            except:
                m = 0
        except:
            s = 0
            m = 0
    except:
        ms = 0
        try:
            m, s = time_str.split('m')
            try:
                s, _ = s.split('s')
            except:
                s = 0
        except:
            m = 0
            s, _ = time_str.split('s')
        
    return float(m)*60 + float(s) + float(ms)*0.001




### main ###

crop = True
ker = 5
depth = 8 
wide = 3
final_gen_plain_predictions(crop, ker, depth, wide)
exit(1)
final_post_process(10, crop, ker, depth, wide)    

## read output timing ##

# prefix = "Total done in "
# prefix = "Conv (with BN) Done in" 
# prefix = "(until CtoS):"
# prefix = "Eval: Eval: ReLU Done in"
# prefix = "Boot (StoC) Done in "
# prefix = "Final (reduce_mean & FC):"
os_path = 'out/out_cr_k5_d1_w2.txt'

for prefix in "Total done in ", "Conv (with BN) Done in" , "(until CtoS):", "Eval: Eval: ReLU Done in", "Boot (StoC) Done in ", "Final (reduce_mean & FC):":
    result_count, result_list = read_out_analysis_time(prefix, os_path)
    for res in result_list:
        print(round(res, prec), end=', ')
    print("\n\n", prefix, result_count, "each", " total iters: ", len(result_list), "mean: ", round(mean(result_list), prec), "std: ", round(stdev(result_list), prec), "min/max: ", min(result_list), "/", max(result_list), "\n\n")
