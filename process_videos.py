import glob
import numpy as np
import cv2
import scipy.io

for fn_mat in sorted(glob.glob('./Aki/LeftPawManual*.mat')):
    print(fn_mat)

    fn_bin = fn_mat[:-3]+'bin'
    fn_avi = './Aki/'+fn_mat[20:-3]+'avi'
    print(fn_avi)

    cap = cv2.VideoCapture(fn_avi)
    ret, frame = cap.read()
    cap.release()



    mat = scipy.io.loadmat(fn_mat)


    headplate_x = np.round(mat['HeadPlatePosition']).astype(np.int)[0][0]
    headplate_y = np.round(mat['HeadPlatePosition']).astype(np.int)[0][1]
    # [y, x] in np.array
    print((headplate_x, headplate_y))




    print(np.max(mat['Left_Paw'][mat['Left_Paw'][:,0]>0,:],axis=0))
    print(np.min(mat['Left_Paw'][mat['Left_Paw'][:,0]>0,:],axis=0))



    offset_x = 100
    offset_y = 80
    center_x = headplate_x+offset_x
    center_y = headplate_y+offset_y
    size_x = 320
    size_y = 160
    print((center_x-size_x//2,center_x+size_x//2))
    print((center_y-size_y//2,center_y+size_y//2))


    with open(fn_bin, 'bw') as f:
        cap = cv2.VideoCapture(fn_avi)
        while(True):
            ret, frame = cap.read()
            if(not ret):
                break
            im = frame[center_y-size_y//2:center_y+size_y//2,center_x-size_x//2:center_x+size_x//2,0]
            f.write(im.tobytes())
        cap.release()
