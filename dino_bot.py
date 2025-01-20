"""
In this script, we demonstrate how to use DINOBot to do one-shot imitation learning.
You first need to install the following repo and its requirements: https://github.com/ShirAmir/dino-vit-features.
You can then run this file inside that repo.

There are a few setup-dependent functions you need to implement, like getting an RGBD observation from the camera
or moving the robot, that you will find on top of this file.
"""
import cv2
import torch
import numpy as np 
import matplotlib.pyplot as plt 
from torchvision import transforms,utils
from PIL import Image
import torchvision.transforms as T
import warnings 
import rospy
import sys
import numpy as np
import open3d as o3d
from spatialmath import SE3

warnings.filterwarnings("ignore")


from my_utils.myImageSaver import MyImageSaver
from my_utils.pose_util import *
from my_utils.myRobotSaver import MyRobotSaver,replay_movement,read_movement
from my_utils.depthUtils import project_to_3d
from my_utils.ransac import ransac
from light_glue_points import find_glue_points,select_object
from mediapipe_hand.read_hand import *
# from track_aruco import get_marker_pose
from my_utils.robot_utils import robot_move,robot_segment_move

sys.path.append('/home/hanglok/work/ur_slam')
from ik_step import init_robot 
import ros_utils.myGripper



#Hyperparameters for DINO correspondences extraction
num_pairs = 10
load_size = 240 #None means no resizing
layer = 9 
facet = 'key' 
bin=True 
thresh=0.05 
model_type='dino_vits8' 
stride=4 
# best_joints = [-0.432199780141012, -0.6554244200335901, 1.9931893348693848, -2.4746230284320276, 2.2496132850646973, -2.539410654698507]
# intrinsics = [ 912.2659912109375, 911.6720581054688, 637.773193359375, 375.817138671875]

intrinsics = [ 447.155, 447.421, 639.532, 359.541]



#Deployment hyperparameters    
ERR_THRESHOLD = 0.02 #A generic error between the two sets of points


#Here are the functions you need to create based on your setup.
def camera_get_rgbd(imagesaver):
    '''
    args:
        imagesaver: class of MyImageSaver
    return:
        rgb: rgb image
        depth: depth image
    '''
    rgb_image = imagesaver.rgb_image
    depth_image = imagesaver.depth_image
    if rgb_image is not None and depth_image is not None:
        imagesaver.record()
        return rgb_image, depth_image
    else:
        raise ValueError("No image received from the camera")

    
def get_hand_curve(filename):
    '''
    "/home/hanglok/work/hand_pose/mediapipe_hand/data_save/norm_point_cloud/2024-11-28_16-12-26.csv"
    '''

    pd_data = hand_pose(filename)
    data = pd_data.get_hand_pose(33, 80)
    back_hand = pd_data.get_back_hand(data)
    SE3_poses = []
    for i in range(len(back_hand)-1):
        R,t = find_transformation(back_hand[i], back_hand[i+1])
        T = np.eye(4)
        T[:3, :3] = R
        t_new = [t_s*0.3 for t_s in t]
        T[:3, 3] = t_new
        SE3_poses.append(T)
    smooth_SE3 = smooth_trajectory(SE3_poses)            
    return smooth_SE3


def move_while_check(imagesaver,mask,rgb_bn,joints_list):
    '''
    check if the robot is moving to the object
    args:
        imagesaver: class of MyImageSaver
        mask: mask of the object
        rgb_bn: bottleneck rgb image
        joints_list: list of joints
    return:
        True: if the error is less than 15
        False: if the error is more than 15
    '''
    for joints in joints_list:
        print("joints",joints)
        robot.move_joints(joints, duration = 0.5, wait=True)
        rgb_check, depth_check = camera_get_rgbd(imagesaver)
        match_point1,match_point2 = find_glue_points(rgb_bn, rgb_check,mask,type="disk")
        if len(match_point1) <= 5 or len(match_point2) <= 5:
            print("no enough points")
            continue
        error = compute_rgb_error(match_point1, match_point2)
        print("error",error)
        if error < 5:
            return True
    return False


def record_demo(robot,robot_name,filename): 
    recorder = MyRobotSaver(robot,robot_name, filename, init_node=False)
    recorder.record_movement()



def replay_demo(robot,filename):
    '''
    args:
        robot: class of robot controller
        filename: string, path to the file containing the recorded movement
    '''

    positions, velocities,transformations = read_movement(filename)
    replay_movement(robot, positions, velocities,transformations,move_to_start=False)


def find_transformation(X, Y):

    # Calculate centroids
    cX = np.mean(X, axis=0)
    cY = np.mean(Y, axis=0)
    # Subtract centroids to obtain centered sets of points
    Xc = X - cX
    Yc = Y - cY
    # Calculate covariance matrix
    C = np.dot(Xc.T, Yc)
    # Compute SVD
    U, S, Vt = np.linalg.svd(C)
    # Determine rotation matrix
    R = np.dot(Vt.T, U.T)
    # Determine translation vector
    t = cY - np.dot(R, cX)
    return R, t

def compute_error(points1, points2):
    return np.linalg.norm(np.array(points1) - np.array(points2))

def compute_rgb_error(match_point1, match_point2):
    '''
    compute the error between the two sets of points in pixel space
    args:
        match_point1: list of points
        match_point2: list of points

    return:
        error: float, the maximum Euclidean distance between the two sets of points
    '''

    # Ensure both match_point1 and match_point2 are converted to numpy arrays
    if isinstance(match_point1, torch.Tensor):
        match_point1 = match_point1.cpu().numpy()
    elif isinstance(match_point1, list):
        match_point1 = np.array([p.cpu().numpy() if isinstance(p, torch.Tensor) else p for p in match_point1])

    if isinstance(match_point2, torch.Tensor):
        match_point2 = match_point2.cpu().numpy()
    elif isinstance(match_point2, list):
        match_point2 = np.array([p.cpu().numpy() if isinstance(p, torch.Tensor) else p for p in match_point2])

    for i in range(len(match_point1)):
        print("match_point1",match_point1[i])
        print("match_point2",match_point2[i])
    
    # Calculate the error as the max Euclidean distance
    error_list = []
    for i in range(len(match_point1)):
        error_list.append(np.linalg.norm(match_point1[i] - match_point2[i]))
    return np.mean(error_list)

def filter_points(points1, points2):
    '''
    Remove points that are [0,0,0]
    '''
    new_points1 = []
    new_points2 = []
    for i in range(len(points1)):
        if sum(points1[i]) != 0 and sum(points2[i]) != 0:
            new_points1.append(points1[i])
            new_points2.append(points2[i])
    return new_points1, new_points2




def object_move_version(robot,imagesaver,intrinsics):
    '''
    let robot to track moving object 
    '''
    torch.cuda.empty_cache()
    rgb_bn, depth_bn = camera_get_rgbd(imagesaver)
    rgb_bn_copy = rgb_bn.copy()
    init_joints =  robot.get_joints() # save current joints
    record_demo(robot,robot_name='robot1',filename='robot1_movements.json')#Record demonstration.
    robot.move_joints(init_joints, duration = 0.5, wait=True)    # move to the initial pose
    # input("Press Enter to continue...")
    masks = select_object(rgb_bn_copy)# choose the object
    error = 100000
    while error > ERR_THRESHOLD:
        # try:
            print("start tracking")
            #Collect observations at the current pose.
            rgb_live, depth_live = camera_get_rgbd(imagesaver)
            # show the image
            # cv2.imshow('RGB image', rgb_live)
            # cv2.waitKey(1)
            count = 0
            point_clouds1 = []
            point_clouds2 = []
            match_point_overall_1 = []
            match_point_overall_2 = []
            while count<5:
                #Compute pixel correspondences between new observation and bottleneck observation.
                match_point1,match_point2 = find_glue_points(rgb_bn, rgb_live,masks,type="disk")
                match_point_overall_1 += match_point1
                match_point_overall_2 += match_point2
                #Given the pixel coordinates of the correspondences, and their depth values,
                #project the points to 3D space
                point_clouds1 += project_to_3d(match_point1, depth_bn, intrinsics,show=False,resize=True ,sequence="xy").copy()
                point_clouds2 += project_to_3d(match_point2,depth_live,intrinsics,show=False,resize=True ,sequence="xy" ).copy()
                count += 1
            origin_points, new_points = filter_points(point_clouds1, point_clouds2)

            # filter outliers
            inliner_mask1 = ransac(origin_points)
            inliner_mask2 = ransac(new_points)
            overall_mask = inliner_mask1 & inliner_mask2
            origin_points = np.array(origin_points)
            new_points = np.array(new_points)
            origin_points = origin_points[overall_mask]
            new_points = new_points[overall_mask]

            #Find rigid translation and rotation that aligns the points by minimising error, using SVD.
            R, t = find_transformation(origin_points,new_points)
            print("moving R: ",R)
            print("moving t: ",t)
    
            error = compute_rgb_error(match_point_overall_1, match_point_overall_2)
            print("Error: ", error)

            if error < 5:
                break
            #Move robot
            joints_list = robot_segment_move(robot,t,R)
            print("joints_list",joints_list)

            if move_while_check(imagesaver,masks,rgb_bn_copy,joints_list):
                break
            torch.cuda.empty_cache()
            # input("Press Enter to continue...")
        # except Exception as e:
        #     print(e,"move to initial pose")
        #     robot.move_joints(init_joints, duration = 0.5, wait=True)
        #     continue
        


    #Once error is small enough, replay demo.
    replay_demo(robot,'records/robot1_movements.json')
    # ADD GRIPPER CONTROL....
    gripper.set_gripper(100,5)
    input("Press Enter to continue...")


# if __name__ == "__main__":
#     rospy.init_node('dino_bot')
#     gripper = ros_utils.myGripper.MyGripper()
#     time.sleep(1)
#     gripper.set_gripper(1000,5)
#     robot = init_robot("robot1",rotate=False)
#     imagesaver = MyImageSaver(cameraNS="camera1")
#     aruco_move(robot,imagesaver)    
    


    
if __name__ == "__main__":
    # object move
    rospy.init_node('dino_bot')
    gripper = ros_utils.myGripper.MyGripper()
    # robot = init_robot("robot1",rotate=False)
    # robot1_ = init_robot("robot1")
    # imagesaver = MyImageSaver(cameraNS="camera1")#     imagesaver = MyImageSaver(cameraNS="camera1")
    # gripper.set_gripper(1000,5)
    # object_move_version(robot,imagesaver,intrinsics)

    pd_data = hand_pose("/home/hanglok/work/hand_pose/mediapipe_hand/data_save/norm_point_cloud/2024-11-28_16-12-26.csv")
    data = pd_data.get_hand_pose(33, 80)
    back_hand = pd_data.get_back_hand(data)
    SE3_poses = []
    for i in range(len(back_hand)-1):
        R,t = find_transformation(back_hand[i], back_hand[i+1])
        T = np.eye(4)
        T[:3, :3] = R
        t_new = [t_s for t_s in t]
        T[:3, 3] = t_new
        SE3_poses.append(T)
    smooth_SE3 = smooth_trajectory(SE3_poses)

    # robot1_.step_in_ee(smooth_SE3,wait =False)
    # gripper.set_gripper(1000,5)


