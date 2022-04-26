import h5py
import numpy as np
import os
import weight_postproc_bn as wpb

# Get the data for ImageNet
def read_ImageNet():
    filename = 'weight_imgnet_ker3_h5/ResNet_18_ker3.h5'
    h5f = h5py.File(filename, 'r')
    cvsfmt = '%.18e'    # covers upto float128
    # get a List of data sets in group 'dd48'
    # print('h5f:', h5f.shape)
    lv0_keys = list(h5f.keys())
    # print("lv0: ", lv0_keys)
    for keys0 in lv0_keys:
        lv1_keys = list(h5f[keys0].keys())
        # print("lv1: ",lv1_keys)
        for keys1 in lv1_keys:
            lv2_keys = list(h5f[keys0][keys1].keys())
            # print("lv2: ",lv2_keys)
            # for keys2 in lv2_keys:
            #     lv3_keys = list(h5f[keys0][keys1][keys2].keys())
                # print("lv3: ",lv3_keys)
            for keys2 in lv2_keys:
                data = h5f[keys0][keys1][keys2]
                # print(data)
                np.savetxt('weight_imgnet_ker3_h5/w-'+str(keys0)+'-'+str(keys1)+'-'+str(keys2)+'-'+str(keys3)+'.csv', np.reshape(data, [-1]), fmt=cvsfmt, delimiter=',')


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

#### Main Start #### 

weight_dir = "Resnet_weights/"
for filename in os.listdir(weight_dir):
    if filename.endswith(".h5"):                        # only read h5 file
        new_filename = filename.replace('nsk_','')      # remove 'nsk_'
        foldername = new_filename.rsplit(".h5")[0]
        out_folder_dir = os.path.join(weight_dir, foldername)
        try:
            os.mkdir(out_folder_dir)          # make folder with weight name
            os.replace(os.path.join(weight_dir, filename), os.path.join(out_folder_dir, new_filename))
            read_ResNet(os.path.join(out_folder_dir, new_filename), out_folder_dir)
            wpb.name_change(out_folder_dir)
            wpb.bn_prep(out_folder_dir)
        except OSError as error:
            print(error)
        except NameError as error:
            print(error, filename)





# read_ResNet()
