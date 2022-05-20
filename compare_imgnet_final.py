import numpy as np
import sys
import os
from statistics import mean, stdev

# for resnet, compare plain with enc
def compare_results(ker, top5):
    max_num_samples = 1000
    num_classes = 1000

    plain_folder_dir = 'ker'+str(ker)+'_data/'
    enc_result_dir = 'ker'+str(ker)+'_enc_result/'

    plain_pred_file = os.path.join(plain_folder_dir, 'plain_prediction_'+str(max_num_samples)+'.csv')
    true_pred_file = os.path.join(plain_folder_dir, 'test_labels_'+str(max_num_samples)+'.csv')
    plain_pred = np.reshape(np.loadtxt(plain_pred_file), [max_num_samples, num_classes])
    true_pred = np.reshape(np.loadtxt(true_pred_file), [max_num_samples])

    acc = 0
    true_acc = 0
    pl_true_acc = 0
    total = 0
    no_iters = []
    wrong_result = {}

    for iter in range(max_num_samples):
        os_path = enc_result_dir+'enc_result_'+str(iter)+'.csv'
        if os.path.exists(os_path):
            read = np.loadtxt(os_path)
            total+=1
        else:
            no_iters.append(iter)
            continue

        res_np = read[:num_classes] #np.reshape(read, [-1])[:10]
        enc_top5 = set(np.argpartition(res_np, -5)[-5:])
        plain_top5 = set(np.argpartition(plain_pred[iter], -5)[-5:])

        if top5:
            if enc_top5 == plain_top5:
                acc += 1
            else:
                wrong_result[str(iter)] = []
                wrong_result[str(iter)].insert(0, enc_top5)
                wrong_result[str(iter)].insert(1, plain_top5)
                wrong_result[str(iter)].insert(2, true_pred[iter])
            if (true_pred[iter] in enc_top5):
                true_acc += 1
            if (true_pred[iter] in plain_top5):
                pl_true_acc += 1
        else:
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
    print("among ", total, " samples.")
    # print("missing: ", no_iters)
    print("\n wrong results: \n")
    for i, result in wrong_result.items():
        if top5:
            print(i, "-th iter.")
            print("enc top 5: ", result[0])
            print("plain top 5: ", result[1], "\n")
            print("true: ", result[2], " \n" )        
        else:
            print(i, "-th iter.")
            print("enc: ", result[0], "argmax: ", np.argmax(result[0]))
            print("plain: ", result[1], "argmax: ", np.argmax(result[1]), "\n")
            print("true: ", result[2], " \n" )

    # tf_images = tf.reshape(tf.constant(np.loadtxt('test_images_'+str(num_samples)+'.csv'), tf.float32), [num_samples, 32, 32, 3])
    # pred = plain_resnet(tf_images)
    # print("enc == plain?", tf.argmax(tf.squeeze(conv, axis=[1,2]),1) == tf.argmax(pred[iter],1))



### main ###

np.set_printoptions(precision=4, threshold=12)
ker = int(sys.argv[1])
compare_results(ker, True)