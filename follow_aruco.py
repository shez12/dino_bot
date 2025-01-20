import re
from flask.cli import F
from gradio_client import file
import rospy
import time
import cv2
import numpy as np
from spatialmath import SE3
import json
import sys
import os
import threading

from aruco_data.id_info import IDInfoList, IDInfo
from my_utils.aruco_util import get_marker_pose
from my_utils.robot_utils import robot_move,robot_fk,robot_ik
from my_utils.myRobotSaver import MyRobotSaver,read_movement,replay_movement
from my_utils.myImageSaver import MyImageSaver
from my_utils.pose_util import matrix_smooth
from my_kalmen_filter import KalmenFilter

sys.path.append('/home/hanglok/work/ur_slam')
from ik_step import init_robot 
import ros_utils.myGripper


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




def save_goal_pose(robot,imagesaver,num_id):
    goal_pose = None
    goal_corner = None

    # Record initial marker pose
    marker_pose = None
    while marker_pose is None:
        frame = imagesaver.rgb_image
        marker_pose, corner = get_marker_pose(frame, num_id)
        print("Still waiting for marker")

    goal_pose = marker_pose.copy()
    goal_corner = corner.copy()
    goal_pose.printline()
    print("goal_corner",goal_corner)
    init_joints = robot.get_joints()  # Save current joints
    filename = f"robot1_aruco_{num_id}.json"
    record_demo(robot, robot_name='robot1', filename=filename)  # Record demonstration
    robot.move_joints(init_joints, duration=0.5, wait=True)
    

    if not os.path.exists(filename):
        # if file not exist, then save the initial pose
        id_info = IDInfo(num_id, goal_pose,goal_corner,filename )
        id_info_list = IDInfoList()
        id_info_list.load_from_json("aruco_data/data.json")
        # delete the old id_info
        num_id = str(num_id)
        if num_id in id_info_list.data:
            del id_info_list.data[num_id]
        id_info_list.add_id_info(id_info)
        id_info_list.save_to_json("aruco_data/data.json")
        print("save initial pose to data.json")



import threading
import cv2
import queue

def aruco_move(robot, imagesaver):
    '''
    Let robot track a moving object using an ArUco marker.
    '''
    num_id = None
    framedelay = 1000 // 20
    init_joints = robot.get_joints()  # Save current joints
    
    # Use a queue to safely pass marker_pose between threads
    marker_pose_queue = queue.Queue()
    filter = KalmenFilter()

    def apply_robot_move(marker_pose, goal_pose):
        move = SE3(marker_pose) * SE3(goal_pose).inv()
        move.printline()
        no_move = SE3(np.eye(4))
        rotation_norm = np.linalg.norm(move.R - np.eye(3))  # Deviation from identity rotation
        translation_norm = np.linalg.norm(move.t)
        print("rotation_norm:", rotation_norm)
        print("translation_norm:", translation_norm)
        if rotation_norm > 0.1 and translation_norm > 0.010:
            robot_move(robot, move.t, move.R)
            print("move r t")
        elif rotation_norm > 0.1:
            robot_move(robot, no_move.t, move.R)
            print("move r")
        elif translation_norm > 0.010:
            robot_move(robot, move.t, no_move.R)
            print("move t")

    # Reset goal pose and corner
    goal_pose = None
    goal_corner = None
    goal_move_file = None
    follow_mode = False

    stop_flag = False
    key = None

    def display_thread():

        nonlocal key, stop_flag
        while not stop_flag:
            frame = imagesaver.rgb_image
            marker_pose, corner = get_marker_pose(frame, num_id)
            

            # Put the marker_pose into the queue so that the main thread can access it
            if marker_pose is not None:
                marker_pose_queue.put(marker_pose)

            if goal_corner is not None and corner is not None:
                for (x1, y1), (x2, y2) in zip(corner, goal_corner):
                    cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 2)

            cv2.imshow('Camera', frame)
            key = cv2.waitKey(framedelay) & 0xFF

        cv2.destroyAllWindows()

    # Start display thread
    display_thread_instance = threading.Thread(target=display_thread)
    display_thread_instance.start()

    try:
        while True:
            # Get the latest marker_pose from the queue
            if not marker_pose_queue.empty():
    # 只取队列中最新的值
                while not marker_pose_queue.empty():
                    marker_pose = marker_pose_queue.get()
                marker_new = True
                filter.new_markerpose(robot_fk(robot,marker_pose))
                filter.Kalman_Filter()
                marker_pose = filter.get_pose()
                print("marker_pose",marker_pose)
                marker_pose = robot_ik(robot,marker_pose)
                # print("marker_pose",marker_pose)
            else:
                marker_pose = None
                marker_new = False


            if key == ord('q'):
                stop_flag = True
                break

            if key == ord('f'):
                follow_mode = not follow_mode
                print("follow_mode", follow_mode)
                time.sleep(2)

            if follow_mode and marker_pose is not None and goal_pose is not None and marker_new:
                apply_robot_move(marker_pose, goal_pose)

            if key == ord('m') and marker_pose is not None and goal_pose is not None and marker_new:
                apply_robot_move(marker_pose, goal_pose)

            if key == ord('c'):
                '''
                set goal object id 
                '''
                num_id = int(input("Enter the number of the object id: "))
                id_info_list = IDInfoList()
                id_info_list.load_from_json("aruco_data/data.json")
                id_info = id_info_list.data[str(num_id)]
                goal_pose = SE3(id_info['pose'])
                goal_corner = np.array(id_info['corner'])

                goal_move_file = id_info['move']
                print("goal_id is", num_id)

            if key == ord('r') and goal_move_file is not None:
                follow_mode = False
                replay_demo(robot, 'records/' + goal_move_file)
                current_joints = robot.get_joints()
                n_count = 0
                while n_count<3:
                    n_count+=1
                    gripper.set_gripper(100, 5)
                    time.sleep(3)
                    replay_demo(robot,"records/robot1_aruco_open.json")
                    gripper.set_gripper(1000, 5)
                    time.sleep(2)
                    robot.move_joints(current_joints, duration=0.5, wait=True)
                gripper.set_gripper(100, 5)
                
            

            if key == ord('b'):
                robot.move_joints(init_joints, duration=0.5, wait=True)
                gripper.set_gripper(1000, 5)

    finally:
        stop_flag = True
        display_thread_instance.join()


if __name__ == "__main__":
    rospy.init_node('dino_bot')
    gripper = ros_utils.myGripper.MyGripper()
    time.sleep(1)
    gripper.set_gripper(1000,5)
    robot = init_robot("robot1",rotate=False)
    imagesaver = MyImageSaver(cameraNS="camera1")
    aruco_move(robot,imagesaver)    
    # save_goal_pose(robot,imagesaver,198)

