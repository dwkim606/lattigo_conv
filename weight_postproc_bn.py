import math
import random
import os
import numpy as np

# rename variables
# process batch normalization, i.e., processing beta, gamma, mean, var => a, b

in_dir = 'weight_h5/'

blocks = [1, 2, 3]
units = [1,2,3]
subs = [1,2]
f_initial = 'w'
f_mid = ''
f_end = ''
f_kinds = ['beta', 'gamma', 'mean', 'var']
f_end_cv = '-conv'


def name_change():
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
                for i in range(len(kinds)):
                    if os.path.exists(in_dir+initial+str(blc)+mid+str(unit)+end+str(sub)+'-'+kinds[i]+'.csv'):
                        os.rename(in_dir+initial+str(blc)+mid+str(unit)+end+str(sub)+'-'+kinds[i]+'.csv', in_dir+f_initial+f_mid+str(num)+'-'+f_kinds[i]+'.csv')
                        conversion = True
                num += 1
    if not conversion:
        print("No conversion for bn.\n")

    # for conv
    conversion = False
    num = 1
    for blc in blocks:
        for unit in units:
            for sub in subs:
                if os.path.exists(in_dir+initial+str(blc)+mid+str(unit)+end_cv+str(sub)+fin_cv+'.csv'):
                    os.rename(in_dir+initial+str(blc)+mid+str(unit)+end_cv+str(sub)+fin_cv+'.csv', in_dir+f_initial+f_mid+str(num)+f_end_cv+'.csv')
                    num += 1
                    conversion = True
    if not conversion:
        print("No conversion for conv.\n")


    # for final
    conversion = False
    for i in range(len(kinds)):
        if os.path.exists(in_dir+'w-final_batchnorm-res_net_cifar10-final_batchnorm-'+kinds[i]+'.csv'):
            os.rename(in_dir+'w-final_batchnorm-res_net_cifar10-final_batchnorm-'+kinds[i]+'.csv', in_dir+'final-'+f_kinds[i]+'.csv')
            conversion = True
    if not conversion:
        print("No conversion for final bn.\n")

    # for initial and final conv
    kindss = ['bias:0', 'kernel:0']
    f_kindss = ['bias', 'kernel']

    conversion = False
    for i in range(len(kindss)):
        if os.path.exists(in_dir+'w-final_conv-res_net_cifar10-final_conv-'+kindss[i]+'.csv'):
            os.rename(in_dir+'w-final_conv-res_net_cifar10-final_conv-'+kindss[i]+'.csv', in_dir+'final-fc'+f_kindss[i]+'.csv')
            conversion = True
    if not conversion:
        print("No conversion for final conv.\n")

    if os.path.exists(in_dir+'w-init_conv-res_net_cifar10-init_conv-kernel:0.csv'):
        os.rename(in_dir+'w-init_conv-res_net_cifar10-init_conv-kernel:0.csv', in_dir+'w0-conv.csv')
        conversion = True
    if not conversion:
        print("No conversion for init conv.\n")



#### Main Start #### 

name_change()

num = 0
for blc in blocks:
    for unit in units:
        for sub in subs:
            beta = np.loadtxt(in_dir+f_initial+f_mid+str(num)+'-beta.csv') 
            gamma = np.loadtxt(in_dir+f_initial+f_mid+str(num)+'-gamma.csv') 
            mean = np.loadtxt(in_dir+f_initial+f_mid+str(num)+'-mean.csv') 
            var = np.loadtxt(in_dir+f_initial+f_mid+str(num)+'-var.csv') 

            bn_a = gamma / np.sqrt(var + 0.001)
            bn_b = beta - mean * gamma / np.sqrt(var + 0.001) 
            
            np.savetxt(in_dir+f_initial+f_mid+str(num)+'-a.csv', bn_a, fmt='%.18e', delimiter=',')
            np.savetxt(in_dir+f_initial+f_mid+str(num)+'-b.csv', bn_b, fmt='%.18e', delimiter=',')
            os.remove(in_dir+f_initial+f_mid+str(num)+'-beta.csv')
            os.remove(in_dir+f_initial+f_mid+str(num)+'-gamma.csv') 
            os.remove(in_dir+f_initial+f_mid+str(num)+'-mean.csv') 
            os.remove(in_dir+f_initial+f_mid+str(num)+'-var.csv') 
            num += 1

beta = np.loadtxt(in_dir+'final-beta.csv') 
gamma = np.loadtxt(in_dir+'final-gamma.csv') 
mean = np.loadtxt(in_dir+'final-mean.csv') 
var = np.loadtxt(in_dir+'final-var.csv') 

bn_a = gamma / np.sqrt(var + 0.001)
bn_b = beta - mean * gamma / np.sqrt(var + 0.001) 

np.savetxt(in_dir+'w'+str(num)+'-a.csv', bn_a, fmt='%.18e', delimiter=',')
np.savetxt(in_dir+'w'+str(num)+'-b.csv', bn_b, fmt='%.18e', delimiter=',')
os.remove(in_dir+'final-beta.csv') 
os.remove(in_dir+'final-gamma.csv') 
os.remove(in_dir+'final-mean.csv') 
os.remove(in_dir+'final-var.csv') 