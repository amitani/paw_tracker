import glob
import os.path
import numpy as np
import cv2
import scipy.io
import struct

for i, fn_mat in enumerate(sorted(glob.glob('./Aki/LeftPawManual*.mat'))):
    fn_bin = fn_mat[:-3]+'bin'
    fn_npy = fn_mat[:-3]+'npy'

files = [open(fn_mat[:-3]+'bin','rb') for fn_mat in sorted(glob.glob('./Aki/LeftPawManual*.mat'))]
npys = [np.load(fn_mat[:-3]+'npy') for fn_mat in sorted(glob.glob('./Aki/LeftPawManual*.mat'))]
frames = [x.shape[0] for x in npys]

for i, fn in enumerate(sorted(glob.glob('./Aki/LeftPawManual*.mat'))):
    print(fn)
    print(frames[i])

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
    def __init__(self, index_list, batch_size, params = []):
        self.rg = np.random.RandomState(47)
        self.index_list = index_list[:]
        self.batch_size = batch_size
        self.params = params
    def getRandomTransform(self, width, height):
        if not self.params:
            return []
        if 'angle' in self.params:
            angle_limit = self.params['angle']
        else:
            angle_limit = 0
        if 'scale' in self.params:
            scale_limit = self.params['scale']
        else:
            scale_limit = 0
        if 'aspect' in self.params:
            aspect_limit = self.params['aspect']
        else:
            aspect_limit = 0
        if 'shift' in self.params:
            shift_limit = self.params['shift']
        else:
            shift_limit = 0

        angle = self.rg.uniform(-angle_limit, angle_limit) # degree
        scale = self.rg.uniform(1 - scale_limit, 1 + scale_limit)
        aspect = self.rg.uniform(1 - aspect_limit, 1 + aspect_limit)
        sx = scale * aspect / (aspect ** 0.5)
        sy = scale / (aspect ** 0.5)
        dx = round(self.rg.uniform(-shift_limit, shift_limit) * width)
        dy = round(self.rg.uniform(-shift_limit, shift_limit) * height)

        cc = np.math.cos(angle / 180 * np.math.pi) * sx
        ss = np.math.sin(angle / 180 * np.math.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)

        return mat

    def get_generator(self):
        def generator():
            while True:
                self.rg.shuffle(self.index_list)
                for start in range(0, len(self.index_list), self.batch_size):
                    x_batch = []
                    y_batch = []
                    end = min(start + self.batch_size, len(self.index_list))
                    ids_train_batch = self.index_list[start:end]
                    for id in ids_train_batch:
                        x, y = read(id[0],id[1])
                        mat = self.getRandomTransform(x.shape[1],x.shape[0])
                        if len(mat)!=0 :
                            #print(mat)
                            #print(y)
                            x = cv2.warpPerspective(x, mat, (x.shape[1], x.shape[0]),
                                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                borderValue=(0, 0, 0))
                            y_tmp = np.ones((3,1))
                            y_tmp[0,0] = y[0]
                            y_tmp[1,0] = y[1]
                            y_tmp = np.dot(mat,y_tmp)
                            y[0] = y_tmp[0,0]/y_tmp[2,0]
                            y[1] = y_tmp[1,0]/y_tmp[2,0]
                            #print(y)
                        x_batch.append(x)
                        y_batch.append(y)
                    x_batch = np.array(x_batch, np.float32) / 256
                    x_batch = x_batch[:,8:-8,16:-16,np.newaxis]
                    x_batch = np.repeat(x_batch, 3, axis=3)
                    y_batch = np.array(y_batch) - np.array([[160, 80]]) # from center
                    y_batch = y_batch
                    yield x_batch, y_batch
        return generator

if __name__ == '__main__':
    i_file = 0
    cv2.namedWindow('image',cv2.WINDOW_NORMAL)
    dg = DataGenerator([[0, x] for x in range(100,250)],1,
        params = {'angle':20,'scale':0.1,'aspect':0.1,'shift':0.2})
    generator = dg.get_generator()
    for i in range(0,100):
        x, y = next(generator())
        print(x.shape)
        print(y.shape)
        x = x[0,:,:]
        y = y[0,:]
        #print(y)
        yi = np.round(y).astype(np.int)
        img = cv2.circle(x,(yi[0]+144,yi[1]+72),5,(0,0,0));
        #print(x)
        cv2.imshow('image',img)
        cv2.waitKey(300)
    cv2.destroyAllWindows()
