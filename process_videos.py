import glob
import os.path
import numpy as np
import cv2
import scipy.io

for fn_mat in sorted(glob.glob('./Aki/LeftPawManual*.mat')):
    print(fn_mat)

    fn_bin = fn_mat[:-3]+'bin'
    fn_npy = fn_mat[:-3]+'npy'
    fn_avi = './Aki/'+fn_mat[20:-3]+'avi'
    print(fn_avi)
    if(os.path.isfile(fn_bin)):
        continue


    cap = cv2.VideoCapture(fn_avi)
    ret, frame = cap.read()
    cap.release()



    mat = scipy.io.loadmat(fn_mat)

    print(np.max(mat['Left_Paw'][mat['Left_Paw'][:,0]>0,:],axis=0))
    print(np.min(mat['Left_Paw'][mat['Left_Paw'][:,0]>0,:],axis=0))
    try:
        headplate_x = np.round(mat['HeadPlatePosition']).astype(np.int)[0][0]
        headplate_y = np.round(mat['HeadPlatePosition']).astype(np.int)[0][1]
        # [y, x] in np.array
        print((headplate_x, headplate_y))
    except:
        headplate_x = 200
        headplate_y = 16

    offset_x = 100
    offset_y = 80
    center_x = headplate_x+offset_x
    center_y = headplate_y+offset_y
    size_x = 320
    size_y = 160
    print((center_x-size_x//2,center_x+size_x//2))
    print((center_y-size_y//2,center_y+size_y//2))

    Y = mat['Left_Paw'][mat['Left_Paw'][:,0]>0,:]
    Y[:,0] = Y[:,0]-(center_x-size_x//2)
    Y[:,1] = Y[:,1]-(center_y-size_y//2)
    np.save(fn_npy,Y)

    frame_count = -1
    with open(fn_bin, 'bw') as f:
        cap = cv2.VideoCapture(fn_avi)
        while(True):
            ret, frame = cap.read()
            if(not ret):
                break
            frame_count+=1
            if(mat['Left_Paw'][frame_count,0]==0):
                continue
            im = frame[center_y-size_y//2:center_y+size_y//2,center_x-size_x//2:center_x+size_x//2,0]
            f.write(im.tobytes())
        cap.release()
