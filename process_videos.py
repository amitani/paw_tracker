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

    mat = scipy.io.loadmat(fn_mat)

    print(np.max(mat['Left_Paw'][mat['Left_Paw'][:,0]>0,:],axis=0))
    print(np.min(mat['Left_Paw'][mat['Left_Paw'][:,0]>0,:],axis=0))
    print(np.median(mat['Left_Paw'][mat['Left_Paw'][:,0]>0,:],axis=0))
    try:
        headplate_x = np.round(mat['HeadPlatePosition']).astype(np.int)[0][0]
        headplate_y = np.round(mat['HeadPlatePosition']).astype(np.int)[0][1]
        # [y, x] in np.array
    except:
        headplate_x = 200
        headplate_y = 16
        print("###HeadPlatePosition Not Found")
    try:
        reference_x = np.round(mat['Reference_pt']).astype(np.int)[0][0]
        reference_y = np.round(mat['Reference_pt']).astype(np.int)[0][1]
        # [y, x] in np.array
    except:
        reference_x = 200
        reference_y = 16
        print("###Reference_pt Not Found")
    print((headplate_x, headplate_y))
    print((reference_x, reference_y))

    offset_x = 60
    offset_y = 50
    center_x = reference_x+offset_x
    center_y = reference_y+offset_y
    size_x = 320
    size_y = 160
    if(center_x < size_x//2):
        center_x = size_x//2
    if(center_y < size_y//2):
        center_y = size_y//2

    print((center_x-size_x//2,center_x+size_x//2))
    print((center_y-size_y//2,center_y+size_y//2))

    if(os.path.isfile(fn_bin) and os.stat(fn_bin).st_size > 0):
        continue


    print('Total annotated frames:')
    print(np.sum(mat['Left_Paw'][:,0]>0))
    I = np.logical_and(mat['Left_Paw'][:,0]>0,mat['Left_Paw'][:,1]<reference_y+150)
    ind = np.nonzero(mat['Left_Paw'][:,0]>0)
    #print(ind)
    #print(ind[0][0])
    #print(ind[0][-1])
    I[ind[0][0]] = 0
    I[ind[0][-1]] = 0
    print('Total valid frames:')
    print(np.sum(I))
    Y = mat['Left_Paw'][I,:]

    Y[:,0] = Y[:,0]-(center_x-size_x//2)
    Y[:,1] = Y[:,1]-(center_y-size_y//2)
    np.save(fn_npy,Y)

    cap = cv2.VideoCapture(fn_avi)
    ret, frame = cap.read()
    cap.release()
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
