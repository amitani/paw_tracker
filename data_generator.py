import glob
import os.path
import numpy as np
import cv2
import scipy.io
import struct

for i, fn_mat in enumerate(sorted(glob.glob('./Aki/LeftPawManual*.mat'))):
    fn_bin = fn_mat[:-3]+'bin'
    fn_npy = fn_mat[:-3]+'npy'

files = [open(fn_mat[:-3]+'bin','rb') for x in sorted(glob.glob('./Aki/LeftPawManual*.mat'))]
npys = [np.load(fn_mat[:-3]+'npy') for x in sorted(glob.glob('./Aki/LeftPawManual*.mat'))]
frames = [x.shape[0] for x in npys]

size_x = 320
size_y = 160
depth = 1
def read(i_file, i_frame):
    files[i_file].seek(size_x*size_y*i_frame*depth)
    x = np.fromfile(files[i_file],np.uint8, size_x*size_y)
    x = x.reshape((size_y,size_x))
    y = npys[i_file][i_frame,:]
    return (x,y)

class DataGenerator:
    rg = np.random.RandomState(47)
    def __init__(self, index_list, batch_size, params = []):
        self.index_list = index_list
        self.batch_size = batch_size
        self.params = params
    def generator(self):
        while True:
                for start in range(0, len(self.index_list), self.batch_size):
                    x_batch = []
                    y_batch = []
                    end = min(start + self.batch_size, len(self.index_list))
                    ids_train_batch = self.index_list[start:end]
                    for id in ids_train_batch:
                        x, y = read(id[0],id[1])
#                        img, mask = randomShiftScaleRotate(img, mask,
#                                                           shift_limit=(-0.0625, 0.0625),
#                                                           scale_limit=(-0.1, 0.1),
#                                                           rotate_limit=(-0, 0))
#                        img, mask = randomHorizontalFlip(img, mask)
#                        mask = np.expand_dims(mask, axis=2)
                        x_batch.append(x)
                        y_batch.append(y)
                    x_batch = np.array(x_batch, np.float32) / 255
                    y_batch = np.array(y_batch)
                    yield x_batch, y_batch

if __name__ == '__main__':
    i_file = 0
    cv2.namedWindow('image',cv2.WINDOW_NORMAL)
    dg = DataGenerator([[0, x] for x in range(100,250)],1)
    generator = dg.generator()
    for i in range(0,100):
        x, y = next(generator)
        #print(x.shape)
        #print(y.shape)
        x = x[0,:,:]
        y = y[0,:]
        #print(y)
        yi = np.round(y).astype(np.int)
        img = cv2.circle(x,(yi[0],yi[1]),5,(255,0,0));
        #print(x)
        cv2.imshow('image',img)
        cv2.waitKey(30)
    cv2.destroyAllWindows()
