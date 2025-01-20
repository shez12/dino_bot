import pyrealsense2 as rs
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2  # Ensure OpenCV is imported

# 初始化管道和滤波器
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.accel)
pipeline.start(config)

# 全局变量存储重力方向
gravity = None

def filter_gravity(accel_data, alpha=0.9):
    """低通滤波器提取重力方向"""
    global gravity
    accel_array = np.array([accel_data.x, accel_data.y, accel_data.z])
    if gravity is None:
        gravity = accel_array
    else:
        gravity = alpha * gravity + (1 - alpha) * accel_array
    return gravity / np.linalg.norm(gravity)

def compute_rotation_matrix(gravity_vector):
    """计算旋转矩阵，将相机坐标系对齐到重力坐标系"""
    z_axis = gravity_vector
    x_axis = np.cross([0, 0, 1], z_axis)
    if np.linalg.norm(x_axis) < 1e-6:  # 处理特殊情况
        x_axis = [1, 0, 0]
    else:
        x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    return np.vstack([x_axis, y_axis, z_axis]).T

def plot_coordinate_system(ax, origin, rotation_matrix, label, color):
    """绘制3D坐标系"""
    axes_length = 0.2  # 坐标轴长度
    x_axis = rotation_matrix[:, 0] * axes_length
    y_axis = rotation_matrix[:, 1] * axes_length
    z_axis = rotation_matrix[:, 2] * axes_length

    ax.quiver(*origin, *x_axis, color=color[0], label=f"{label} X-axis", arrow_length_ratio=0.1)
    ax.quiver(*origin, *y_axis, color=color[1], label=f"{label} Y-axis", arrow_length_ratio=0.1)
    ax.quiver(*origin, *z_axis, color=color[2], label=f"{label} Z-axis", arrow_length_ratio=0.1)

try:
    # 初始化Matplotlib 3D绘图
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    while True:
        # 获取帧数据
        frames = pipeline.wait_for_frames()
        accel_frame = frames.first_or_default(rs.stream.accel)

        if accel_frame:
            # 提取加速度计数据
            accel_data = accel_frame.as_motion_frame().get_motion_data()
            gravity_vector = filter_gravity(accel_data)

            # 计算重力坐标系旋转矩阵
            rotation_matrix = compute_rotation_matrix(gravity_vector)

            # 清除旧的绘图
            ax.cla()
            ax.set_xlim([-1, 1])
            ax.set_ylim([-1, 1])
            ax.set_zlim([-1, 1])
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')

            # 绘制重力坐标系
            plot_coordinate_system(ax, [0, 0, 0], np.eye(3), "Gravity", ['r', 'g', 'b'])
            
            # 绘制相机坐标系
            plot_coordinate_system(ax, [0.5, 0.5, 0.5], rotation_matrix, "Camera", ['c', 'm', 'y'])

            # 显示图例
            ax.legend()

            # 更新绘图
            plt.draw()
            plt.pause(0.01)


finally:
    # 停止管道
    pipeline.stop()
    print("数据采集已停止。")
    cv2.destroyAllWindows()  # Close any OpenCV windows
