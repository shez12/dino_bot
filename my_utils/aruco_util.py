import cv2
import cv2.aruco as aruco
import numpy as np
from my_utils.pose_util import Rt_to_pose,inverse_pose, pose_to_SE3
from my_utils.camera_intrinsics import intrinsics

ARUCO_DICT_NAME = aruco.DICT_4X4_250
# ARUCO_DICT_NAME = aruco.DICT_APRILTAG_36H11
my_aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_NAME)

# Function to detect ArUco markers

def detect_aruco(image, draw_flag=False):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detector = aruco.ArucoDetector(my_aruco_dict, aruco.DetectorParameters())
    corners, ids, rejectedImgPoints = detector.detectMarkers(gray)
    
    if corners is None or ids is None:
        return [], []
    else:
        # Refine corners to subpixel accuracy
        refined_corners = []
        for corner in corners:
            refined_corner = cv2.cornerSubPix(
                gray,  # Gray image
                corner.astype(np.float32),  # Initial corners
                winSize=(5, 5),  # Larger window size for better accuracy
                zeroZone=(-1, -1),  # Default: no dead zone
                criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.0001)  # Higher precision and more iterations
            )
            refined_corners.append(refined_corner)
        
        # Draw detected markers on the image
        if draw_flag and ids is not None:
            image = aruco.drawDetectedMarkers(image, refined_corners, ids)  
        
        # Flatten the refined corners for compatibility
        refined_corners = [c.reshape(-1, 2) for c in refined_corners]
        ids = ids.flatten().tolist()
        return refined_corners, ids




# Function to generate ArUco markers
def generate_aruco_marker(marker_id, marker_size, output_file):
    # Generate ArUco marker image
    marker_image = np.zeros((marker_size, marker_size), dtype=np.uint8)
    marker_image = aruco.generateImageMarker(my_aruco_dict, marker_id, marker_size, marker_image, 1)
    cv2.imwrite(output_file, marker_image)


def estimate_markers_poses(corners, marker_size, intrinsics,frame):
    '''
    This will estimate the rvec and tvec for each of the marker corners detected by:
       corners, ids, rejectedImgPoints = detector.detectMarkers(image)
    corners - is an array of detected corners for each detected marker in the image
    marker_size - is the size of the detected markers
    mtx - is the camera matrix
    distortion - is the camera distortion matrix
    '''
    # make sure the aruco's orientation in the camera view! 
    marker_points = np.array([[-marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, -marker_size / 2, 0],
                              [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)
    
    mtx = np.array([[intrinsics["fx"], 0, intrinsics["cx"]],
                    [0, intrinsics["fy"], intrinsics["cy"]],
                    [0, 0, 1]], dtype=np.float32)
    # distortion = np.zeros((5, 1))  # Assuming no distortion
    distortion  = np.array([[ 0.00377581 , 0.00568285 ,-0.00188039, -0.00102468 , 0.02337337]])

    poses = []
    for c in corners:
        ret, rvec, tvec = cv2.solvePnP(marker_points, c, mtx, distortion)
        if ret:
            tvec = tvec.reshape((3,))
            R, _ = cv2.Rodrigues(rvec)
            pose = Rt_to_pose(R, tvec)  # Ensure Rt_to_pose is correctly implemented
            poses.append(pose)
            cv2.drawFrameAxes(frame, mtx, distortion, rvec, tvec, marker_size)
        else:
            print("Pose estimation failed for one of the markers")
    return poses


def get_aruco_poses(corners, ids, intrinsics,frame):
    # make sure the aruco's orientation in the camera view! 
    poses = estimate_markers_poses(corners, marker_size=0.02, intrinsics=intrinsics,frame=frame)  # Marker size in meters
    poses_dict = {}
    # detected
    if ids is not None:
        for k, iden in enumerate(ids):
            poses_dict[iden]=poses[k] 
    return poses_dict


def get_cam_pose(frame, intrinsics):
    corners, ids = detect_aruco(frame, draw_flag=True)# 
    poses_dict = get_aruco_poses(corners=corners, ids=ids, intrinsics=intrinsics,frame=frame)
    id = 0
    current_cam = None
    if id in poses_dict:
        current_pose = poses_dict[id]
            # compute the R, t
        current_cam = inverse_pose(current_pose)
            # compute 
        # print('cam', np.round(current_cam[:3], 3))
    return current_cam

def get_marker_pose(frame, id=0, draw=True):
    corners, ids = detect_aruco(frame, draw_flag=draw)# 
    if ids is not None and len(ids)>0:
        poses_dict = get_aruco_poses(corners=corners, ids=ids, intrinsics=intrinsics,frame=frame)
        for i, c in zip(ids, corners):
            if i == id:
                pose = pose_to_SE3(poses_dict[id])
                return pose, c
    return None, None

    