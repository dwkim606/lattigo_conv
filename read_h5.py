import h5py
import numpy as np
filename = 'weights_noskip.h5'
h5f = h5py.File(filename, 'r')
cvsfmt = '%.18e'    # covers upto float128
# get a List of data sets in group 'dd48'
# print('h5f:', h5f.shape)

# Get the data
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
                # print(data)
                np.savetxt('weight_h5/w-'+str(keys0)+'-'+str(keys1)+'-'+str(keys2)+'-'+str(keys3)+'.csv', np.reshape(data, [-1]), fmt=cvsfmt, delimiter=',')