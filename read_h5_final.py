import h5py
import numpy as np
import os

# rename variables
# process batch normalization, i.e., processing beta, gamma, mean, var => a, b

blocks = [1,2,3]
units = [1,2,3]
subs = [1,2]
f_initial = 'w'
f_mid = ''
f_end = ''
f_kinds = ['beta', 'gamma', 'mean', 'var']
f_end_cv = '-conv'

def name_change(in_dir):
    # in_dir = 'weight_ker3_crop_h5/'#'weight_h5/'
    kinds = ['beta:0', 'gamma:0', 'moving_mean:0', 'moving_variance:0']
    initial = 'w-block'
    mid = '-res_net_unit_'
    end = '-batchnorm_'
    end_cv = '-conv'
    fin_cv = '-kernel:0'

    # for bn
    conversion = False
    num = 0
    for blc in blocks:
        for unit in units:
            for sub in subs:
                if os.path.exists(os.path.join(in_dir,initial+str(blc)+mid+str(unit)+end+str(sub)+'-'+kinds[0]+'.csv')):
                    for i in range(len(kinds)):
                        os.rename(os.path.join(in_dir,initial+str(blc)+mid+str(unit)+end+str(sub)+'-'+kinds[i]+'.csv'), os.path.join(in_dir,f_initial+f_mid+str(num)+'-'+f_kinds[i]+'.csv'))
                        conversion = True
                    num += 1
    if not conversion:
        raise NameError('Err: No conversion for bn.\n')
        # print("No conversion for bn.\n")

    # for conv
    conversion = False
    num = 1
    for blc in blocks:
        for unit in units:
            for sub in subs:
                if os.path.exists(os.path.join(in_dir,initial+str(blc)+mid+str(unit)+end_cv+str(sub)+fin_cv+'.csv')):
                    os.rename(os.path.join(in_dir,initial+str(blc)+mid+str(unit)+end_cv+str(sub)+fin_cv+'.csv'), os.path.join(in_dir,f_initial+f_mid+str(num)+f_end_cv+'.csv'))
                    num += 1
                    conversion = True
    if not conversion:
        raise NameError('No conversion for conv.\n')
        # print("No conversion for conv.\n")


    # for final
    conversion = False
    for i in range(len(kinds)):
        if os.path.exists(os.path.join(in_dir,'w-final_batchnorm-res_net_cifar10-final_batchnorm-'+kinds[i]+'.csv')):
            os.rename(os.path.join(in_dir,'w-final_batchnorm-res_net_cifar10-final_batchnorm-'+kinds[i]+'.csv'), os.path.join(in_dir,'final-'+f_kinds[i]+'.csv'))
            conversion = True
    if not conversion:
        raise NameError('No conversion for final bn.\n')
        # print("No conversion for final bn.\n")

    # for initial and final conv
    kindss = ['bias:0', 'kernel:0']
    f_kindss = ['bias', 'kernel']

    conversion = False
    for i in range(len(kindss)):
        if os.path.exists(os.path.join(in_dir,'w-final_conv-res_net_cifar10-final_conv-'+kindss[i]+'.csv')):
            os.rename(os.path.join(in_dir,'w-final_conv-res_net_cifar10-final_conv-'+kindss[i]+'.csv'), os.path.join(in_dir,'final-fc'+f_kindss[i]+'.csv'))
            conversion = True
    if not conversion:
        raise NameError('No conversion for final conv.\n')
        # print("No conversion for final conv.\n")

    if os.path.exists(os.path.join(in_dir,'w-init_conv-res_net_cifar10-init_conv-kernel:0.csv')):
        os.rename(os.path.join(in_dir,'w-init_conv-res_net_cifar10-init_conv-kernel:0.csv'), os.path.join(in_dir,'w0-conv.csv'))
        conversion = True
    if not conversion:
        raise NameError('No conversion for init conv.\n')
        # print("No conversion for init conv.\n")

def bn_prep(in_dir):
    num = 0
    while os.path.exists(os.path.join(in_dir,f_initial+f_mid+str(num)+'-beta.csv')):
        beta = np.loadtxt(os.path.join(in_dir,f_initial+f_mid+str(num)+'-beta.csv'))
        gamma = np.loadtxt(os.path.join(in_dir,f_initial+f_mid+str(num)+'-gamma.csv'))
        mean = np.loadtxt(os.path.join(in_dir,f_initial+f_mid+str(num)+'-mean.csv'))
        var = np.loadtxt(os.path.join(in_dir,f_initial+f_mid+str(num)+'-var.csv'))

        bn_a = gamma / np.sqrt(var + 0.001)
        bn_b = beta - mean * gamma / np.sqrt(var + 0.001) 
        
        np.savetxt(os.path.join(in_dir,f_initial+f_mid+str(num)+'-a.csv'), bn_a, fmt='%.18e', delimiter=',')
        np.savetxt(os.path.join(in_dir,f_initial+f_mid+str(num)+'-b.csv'), bn_b, fmt='%.18e', delimiter=',')
        os.remove(os.path.join(in_dir,f_initial+f_mid+str(num)+'-beta.csv'))
        os.remove(os.path.join(in_dir,f_initial+f_mid+str(num)+'-gamma.csv'))
        os.remove(os.path.join(in_dir,f_initial+f_mid+str(num)+'-mean.csv'))
        os.remove(os.path.join(in_dir,f_initial+f_mid+str(num)+'-var.csv'))
        num += 1

    beta = np.loadtxt(os.path.join(in_dir,'final-beta.csv'))
    gamma = np.loadtxt(os.path.join(in_dir,'final-gamma.csv'))
    mean = np.loadtxt(os.path.join(in_dir,'final-mean.csv'))
    var = np.loadtxt(os.path.join(in_dir,'final-var.csv'))

    bn_a = gamma / np.sqrt(var + 0.001)
    bn_b = beta - mean * gamma / np.sqrt(var + 0.001) 

    np.savetxt(os.path.join(in_dir,'w'+str(num)+'-a.csv'), bn_a, fmt='%.18e', delimiter=',')
    np.savetxt(os.path.join(in_dir,'w'+str(num)+'-b.csv'), bn_b, fmt='%.18e', delimiter=',')
    os.remove(os.path.join(in_dir,'final-beta.csv'))
    os.remove(os.path.join(in_dir,'final-gamma.csv')) 
    os.remove(os.path.join(in_dir,'final-mean.csv'))
    os.remove(os.path.join(in_dir,'final-var.csv'))


# Get the data for ResNet
def read_ResNet(filename, out_dir):
    # filename = 'weight_ker7_crop_wide_h5/weights_nsk_crop_ker7_wide32.h5'
    h5f = h5py.File(filename, 'r')
    cvsfmt = '%.18e'    # covers upto float128

    lv0_keys = list(h5f.keys())
    # print("lv0: ", lv0_keys)
    for keys0 in lv0_keys:
        lv1_keys = list(h5f[keys0].keys())
        # print("lv1: ",lv1_keys)
        for keys1 in lv1_keys:
            lv2_keys = list(h5f[keys0][keys1].keys())
            # print("lv2: ",lv2_keys)
            for keys2 in lv2_keys:
                lv3_keys = list(h5f[keys0][keys1][keys2].keys())
                # print("lv3: ",lv3_keys)
                for keys3 in lv3_keys:
                    data = h5f[keys0][keys1][keys2][keys3]
                    np.savetxt(os.path.join(out_dir,'w-'+str(keys0)+'-'+str(keys1)+'-'+str(keys2)+'-'+str(keys3)+'.csv'), np.reshape(data, [-1]), fmt=cvsfmt, delimiter=',')

def gen_enc_result_folders():
    weight_dir = "Resnet_weights/"
    out_dir = "Resnet_enc_results/"
    for filename in os.listdir(weight_dir):
        new_filename = "results"+filename.lstrip("weights")
        try:
            os.mkdir(os.path.join(out_dir, new_filename))        
        except OSError as error:
            print(error)


#### Main Start #### 

# read h5 files
# convert into csv files and save into folder with the same name
# rename it for plain python and lattigo

weight_dir = "Resnet_weights/"
for filename in os.listdir(weight_dir):
    if filename.endswith(".h5"):                        # only read h5 file
        foldername = filename.replace('nsk_','').replace('cf100_','',1).rsplit('_iter')[0]
        new_filename = foldername+".h5"
        out_folder_dir = os.path.join(weight_dir, foldername)
        try:
            os.mkdir(out_folder_dir)          # make folder with weight name
            os.replace(os.path.join(weight_dir, filename), os.path.join(out_folder_dir, new_filename))
            read_ResNet(os.path.join(out_folder_dir, new_filename), out_folder_dir)
            name_change(out_folder_dir)     # change weights name
            bn_prep(out_folder_dir)         # bn weights into a and b
        except OSError as error:
            print(error)
        except NameError as error:
            print(error, filename)

gen_enc_result_folders()




# read_ResNet()
