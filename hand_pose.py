import rospy
import cv2
import mediapipe as mp
import sys
import numpy as np
import time



sys.path.append('/home/hanglok/work/ur_slam')
from ros_utils.myImageSaver import MyImageSaver
from my_utils.depthUtils import project_to_3d

sys.path.append("/home/hanglok/work/Depth-Anything-V2")
from depth_anything import depth_anything, model_init

# import csv
intrinsics = [ 912.2659912109375, 911.6720581054688, 637.773193359375, 375.817138671875]



if __name__ == "__main__":
    rospy.init_node('hand_pose')
    image_saver = MyImageSaver(cameraNS='camera1')
    rospy.sleep(1)
    framedelay = 1000 // 20
    model = model_init()
    goal_frame = None
    src_pts = None
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5, max_num_hands=2)
    mp_drawing = mp.solutions.drawing_utils
    landmark_array = []  # Initialize an empty list to store landmark data
    save_points = []
    while not rospy.is_shutdown():
        frame = image_saver.rgb_image
        show_frame = frame.copy()  # In case of frame is ruined by painting frame
        depth = depth_anything(frame,model)

        # Convert the BGR image to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)



        # print("frame_rgb",frame_rgb.shape)
        # print("depth",depth.shape)

        # Process the frame and extract hand landmarks
        results = hands.process(frame_rgb)
        points =[]
        if results.multi_hand_landmarks:
            # Collect landmarks for the current frame
            landmark_seq = []
            for hand_landmarks in results.multi_hand_landmarks:#multi_hand_world_landmarks
                for landmark in hand_landmarks.landmark:
                    landmark_seq.append([landmark.x, landmark.y, landmark.z])
                    #find pixel coordinate
                    x = int(round(landmark.x * frame_rgb.shape[1]))
                    y = int(round(landmark.y * frame_rgb.shape[0]))
                    #find depth value
                    points.append([y,x])
            #calculated 3d points
            points_3d = project_to_3d(points=points, depth=depth, intrinsics=intrinsics, show=False,resize=False,sequence="yx")
            save_points.append(np.array(points_3d))
            print("3d point", points_3d)
            print("////////////////////////////////")
            # Convert to numpy array and append to landmark_array
            # landmark_array.append(np.array(landmark_seq))
            save_points.append(np.array([-1,-1,-1]).reshape(1, -1)) 
            # landmark_array.append(np.array([-1,-1,-1]).reshape(1, -1))  # Reshape to 2D array


            # Draw landmarks on the frame    
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(show_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        # Display the frame with landmarks
        cv2.imshow('Hand Pose Detection', show_frame)

        # Handle key events (optional)
        if cv2.waitKey(framedelay) & 0xFF == ord('q'):
            break


    # Clean up
    cv2.destroyAllWindows()

    # Convert landmark_array to a single numpy array before saving
    filename = 'hand_pose/'+'landmarks'+str(time.time())+'.npy'
    if save_points:
        save_points = np.concatenate(save_points, axis=0)  # Concatenate all frames into a single array
        np.save(filename, save_points)  # Save to a single .npy file
        print(f"Landmarks saved to {filename}")
