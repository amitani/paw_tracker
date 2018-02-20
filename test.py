import glob
import os.path
import numpy as np
import cv2
import scipy.io
import struct

size_x = 320
size_y = 160
cv2.namedWindow('image',cv2.WINDOW_NORMAL)
for fn_bin in sorted(glob.glob('./Aki/*.bin')):
    print(fn_bin)
    fn_npy = fn_bin[:-3]+'npy'
    Y = np.round(np.load(fn_npy)).astype(np.int)
    frame_count = -1
    with open(fn_bin, 'br') as f:
        while True:
            im = np.fromfile(f,np.uint8,size_y*size_x)
            if(im.size<size_y*size_x):
                break
            im = np.reshape(im,(size_y,size_x))
            frame_count += 1
            im = np.repeat(im[:,:,np.newaxis],3,axis=2)
            cv2.circle(im, (Y[frame_count,0], Y[frame_count,1]),5,(255,0,0))
            cv2.imshow('image',im)
            cv2.waitKey(10)
