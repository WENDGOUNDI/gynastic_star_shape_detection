# gynastic_star_shape_detection
Here, we are detecting the "star" gymnastic figure using a Deep Learning based Human Pose Estimation with OpenCV. Human Pose Estimation is often used to detect keypoint locations that describe the body position or the excecuted movement: standing, sitting or running to cite few. The pretrained model used here is based on MPII Human Pose Dataset, offering 15 distincts points: Head – 0, Neck – 1, Right Shoulder – 2, Right Elbow – 3, Right Wrist – 4, Left Shoulder – 5, Left Elbow – 6, Left Wrist – 7, Right Hip – 8, Right Knee – 9, Right Ankle – 10, Left Hip – 11, Left Knee – 12, Left Ankle – 13, Chest – 14, Background – 15.

![kepoints_ MPI](https://user-images.githubusercontent.com/48753146/152742404-ea95c5ef-fd31-4567-97b8-12458e9070b3.png)

# Download Weights
We are using a model trained on Caffe Deep Learning Framework. We need two files:
 - Prototxt file (pose_deploy_linevec_faster_4_stages.prototxt): this file describes the network architecture.
 - caffemodel file (pose_iter_160000.caffemodel): it stores the weights of the trained model.
The above two files should be placed in the weights folder to rn the code successfully.

# Procedure
 - Load the pre-trained models
 - Read and prepare the input image for the network
 - Detect body points and Calculating pose estimation
 - Apply arithmetic comparison to detect the desire gesture. In our case, we will focus on right_wrist, right_elbow, left_wrist, left_elbow, left_ankle, left_hip, right_ankle and right_hip to detect the "star" figure.

## display keypoints 
![image_kepypoints](https://user-images.githubusercontent.com/48753146/152743300-0e8da087-d97f-4ad5-9d67-d8ac8c394ec6.jpg)

References:
pose_iter_160000.caffemodel weight can be downloaded through this link https://pythonwife.com/files/cv/pose_iter_160000.caffemodel .
