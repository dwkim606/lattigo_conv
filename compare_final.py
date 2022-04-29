import numpy as np
import sys
import os
from statistics import mean, stdev

# for resnet, compare plain with enc
def compare_results(iter_num, crop, ker, depth, wid):    
    max_num_samples = 1000

    if crop:
        plain_pred_file = 'Resnet_plain_data/plain_prediction_crop_ker'+str(ker)+'_d'+str(depth)+'_wid'+str(wid)+'_'+str(max_num_samples)+'.csv'
        enc_result_dir = 'Resnet_enc_results/results_crop_ker'+str(ker)+'_d'+str(depth)+'_wid'+str(wid)+'/'
    else:
        plain_pred_file = 'Resnet_plain_data/plain_prediction_ker'+str(ker)+'_d'+str(depth)+'_wid'+str(wid)+'_'+str(max_num_samples)+'.csv'
        enc_result_dir = 'Resnet_enc_results/results_ker'+str(ker)+'_d'+str(depth)+'_wid'+str(wid)+'/'

    plain_pred = np.reshape(np.loadtxt(plain_pred_file), [max_num_samples, 10])    
    true_pred = np.reshape(np.loadtxt('Resnet_plain_data/test_labels_'+str(max_num_samples)+'.csv'), [max_num_samples])    

    acc = 0
    true_acc = 0
    pl_true_acc = 0
    total = 0
    no_iters = []
    wrong_result = {}
    os_path = enc_result_dir+'class_result_ker'+str(ker)+'_'

    for iter in range(iter_num):
        if os.path.exists(os_path+str(iter)+'.csv'):
            read = np.loadtxt(os_path+str(iter)+'.csv')
            total+=1
        else:
            no_iters.append(iter)
            continue

        res_np = read[:10] #np.reshape(read, [-1])[:10]
        # print("enc: ", res_np, "argmax: ", np.argmax(res_np))
        # print("plain: ", plain_pred[iter], "argmax: ", np.argmax(plain_pred[iter]))
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


### main ###

crop = True
num_iter = 100
ker = int(sys.argv[1])
depth = int(sys.argv[2])
wide = int(sys.argv[3])

compare_results(num_iter, crop, ker, depth, wide)

