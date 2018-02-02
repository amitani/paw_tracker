LeftPawManual_MD009_170808_original.mat matches the video file 'MD009_170808.avi'. 
For this video, Maddy tracked the full video for the left forepaw in the side view and ~10,000 frames in the mirror view. 

LeftPawManual_MD010_170808_original.mat matches the video file 'MD010_170808.avi'
She did the first ~10,000 frames of both side and mirror views for the left forepaw. 

In the files:
Left_Paw is the side view pixel location. 
Left_Paw_Mirror is the bottom view. 
they are frame x 2 matrices, indicating paw location in the image.
There are a lot of zeros, which are frames the paw is not moving or after 10,000 frames (see above). 
HeadPlatePosition is the location of the reference point we choose across videos/days. 