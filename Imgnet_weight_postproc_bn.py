import math
import random
import os
import numpy as np

# rename variables
# process batch normalization, i.e., processing beta, gamma, mean, var => a, b

in_dir = 'weight_imgnet_ker5_h5/'

blocks = [0, 1, 2, 3, 4]
units = [0, 1]
subs = [0, 1]
f_initial = 'w'
f_kinds = ['beta', 'gamma', 'mean', 'var']
f_end_cv = '-conv'

# must modify so that bn is before the average pool
def name_change():
    kinds = ['beta:0', 'gamma:0', 'moving_mean:0', 'moving_variance:0']
    initial = 'w-conv'
    end = '-bn_'
    end_cv = '-conv'
    fin_cv = '-kernel:0'

    # for bn
    conversion = False
    num = 1
    for blc in blocks:
        for unit in units:
            for sub in subs:
                for i in range(len(kinds)):
                    src = in_dir+initial+str(blc)+'_'+str(unit)+'-conv'+str(blc)+'_'+str(unit)+end+str(sub)+'-'+kinds[i]+'.csv'
                    des = in_dir+f_initial+str(num)+'-'+f_kinds[i]+'.csv'
                    if os.path.exists(src):
                        os.rename(src, des)
                        conversion = True
                    else:
                        conversion = False
                if conversion:
                    num += 1
    print(num, "bn conversion done.")
    
    # for conv
    conversion = False
    end = '-conv_'
    num = 1
    for blc in blocks:
        for unit in units:
            for sub in subs:
                src = in_dir+initial+str(blc)+'_'+str(unit)+'-conv'+str(blc)+'_'+str(unit)+end+str(sub)+'-kernel:0.csv'
                des = in_dir+f_initial+str(num)+'-conv'+'.csv'
                if os.path.exists(src):
                    os.rename(src, des)
                    conversion = True
                else:
                    conversion = False
                if conversion:
                    num += 1
    print(num, "conv conversion done.")

def bn_computation(iters):
    for num in range(1, iters):
        beta = np.loadtxt(in_dir+f_initial+str(num)+'-beta.csv') 
        gamma = np.loadtxt(in_dir+f_initial+str(num)+'-gamma.csv') 
        mean = np.loadtxt(in_dir+f_initial+str(num)+'-mean.csv') 
        var = np.loadtxt(in_dir+f_initial+str(num)+'-var.csv') 

        bn_a = gamma / np.sqrt(var + 0.001)
        bn_b = beta - mean * gamma / np.sqrt(var + 0.001) 
        
        np.savetxt(in_dir+f_initial+str(num)+'-a.csv', bn_a, fmt='%.18e', delimiter=',')
        np.savetxt(in_dir+f_initial+str(num)+'-b.csv', bn_b, fmt='%.18e', delimiter=',')
        os.remove(in_dir+f_initial+str(num)+'-beta.csv')
        os.remove(in_dir+f_initial+str(num)+'-gamma.csv') 
        os.remove(in_dir+f_initial+str(num)+'-mean.csv') 
        os.remove(in_dir+f_initial+str(num)+'-var.csv') 


#### Main Start #### 

# name_change()
# bn_computation(14)
