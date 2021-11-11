import matplotlib.pyplot as plt
import os
import numpy as np
import sys

ID = int(sys.argv[1])

width = 32;
result = np.reshape(np.loadtxt('DCGAN_result/result_{:04d}.txt'.format(ID)), [width, width])
result = np.tanh(result)
plt.imshow(result * 127.5 + 127.5, cmap='gray')
plt.savefig('enc_image_{:04d}.png'.format(ID))

#plt.savefig(os.path.join('./samples', 'image_{:04d}.png'.format(ID)))
