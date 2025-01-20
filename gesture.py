import math
import numpy as np

"""
计算两个点之间的距离：L2距离（欧式距离）
"""
def points_distance(x0, y0, x1, y1):
    return math.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)


"""
计算两条线段之间的夹角，以弧度表示
"""
def compute_angle(x0, y0, x1, y1, x2, y2, x3, y3):
    AB = [x1 - x0, y1 - y0]
    CD = [x3 - x2, y3 - y2]

    dot_product = AB[0] * CD[0] + AB[1] * CD[1]

    AB_distance = points_distance(x0, y0, x1, y1) + 0.001   # 防止分母出现0
    CD_distance = points_distance(x2, y2, x3, y3) + 0.001

    cos_theta = dot_product / (AB_distance * CD_distance)

    theta = math.acos(cos_theta)

    return theta


"""
检测所有手指状态（判断每根手指收紧 or 伸直）
大拇指只有收紧和伸直两种状态，其他手指除了收紧和伸直还包含第三种状态（手指没有伸直，但是也没有达到收紧的标准）
"""
def detect_all_finger_state(data):

    finger_first_angle_bend_threshold = math.pi * 0.25  # 大拇指收紧阈值
    finger_other_angle_bend_threshold = math.pi * 0.5  # 其他手指收紧阈值
    finger_other_angle_straighten_threshold = math.pi * 0.2  # 其他手指伸直阈值

    first_is_bend = False
    first_is_straighten = False
    second_is_bend = False
    second_is_straighten = False
    third_is_bend = False
    third_is_straighten = False
    fourth_is_bend = False
    fourth_is_straighten = False
    fifth_is_bend = False
    fifth_is_straighten = False

    finger_first_angle = compute_angle(data[0][0], data[0][1], data[1][0], data[1][1],
                                        data[2][0], data[2][1], data[4][0], data[4][1])
    finger_sencond_angle = compute_angle(data[0][0], data[0][1], data[5][0], data[5][1],
                                        data[6][0], data[6][1], data[8][0], data[8][1])
    finger_third_angle = compute_angle(data[0][0], data[0][1], data[9][0], data[9][1],
                                        data[10][0], data[10][1], data[12][0], data[12][1])
    finger_fourth_angle = compute_angle(data[0][0], data[0][1], data[13][0], data[13][1],
                                        data[14][0], data[14][1], data[16][0], data[16][1])
    finger_fifth_angle = compute_angle(data[0][0], data[0][1], data[17][0], data[17][1],
                                        data[18][0], data[18][1], data[20][0], data[20][1])

    if finger_first_angle > finger_first_angle_bend_threshold:              # 判断大拇指是否收紧
        first_is_bend = True
        first_is_straighten = False
    else:
        first_is_bend = False
        first_is_straighten = True

    if finger_sencond_angle > finger_other_angle_bend_threshold:            # 判断食指是否收紧
        second_is_bend = True
    elif finger_sencond_angle < finger_other_angle_straighten_threshold:
        second_is_straighten = True
    else:
        second_is_bend = False
        second_is_straighten = False

    if finger_third_angle > finger_other_angle_bend_threshold:              # 判断中指是否收紧
        third_is_bend = True
    elif finger_third_angle < finger_other_angle_straighten_threshold:
        third_is_straighten = True
    else:
        third_is_bend = False
        third_is_straighten = False

    if finger_fourth_angle > finger_other_angle_bend_threshold:             # 判断无名指是否收紧
        fourth_is_bend = True
    elif finger_fourth_angle < finger_other_angle_straighten_threshold:
        fourth_is_straighten = True
    else:
        fourth_is_bend = False
        fourth_is_straighten = False

    if finger_fifth_angle > finger_other_angle_bend_threshold:              # 判断小拇指是否收紧
        fifth_is_bend = True
    elif finger_fifth_angle < finger_other_angle_straighten_threshold:
        fifth_is_straighten = True
    else:
        fifth_is_bend = False
        fifth_is_straighten = False

    # 将手指的收紧或伸直状态存在字典中，简化后续函数的参数
    bend_states = {'first': first_is_bend, 'second': second_is_bend, 'third': third_is_bend, 'fourth': fourth_is_bend, 'fifth': fifth_is_bend}
    straighten_states = {'first': first_is_straighten, 'second': second_is_straighten, 'third': third_is_straighten, 'fourth': fourth_is_straighten, 'fifth': fifth_is_straighten}

    return bend_states, straighten_states


"""
判断是否为读取数据的手势
读取数据判断条件：
1：大拇指不能为收紧
2：其他手指至少有一根不能收紧
3：待商榷


现在条件：
1：大拇指不能收紧
2：食指和中止在收紧和伸直之间
"""
def judge_data(bend_states, straighten_states,show = "False"):
    if show == "True":
        print('straighten_states,first:'+str(straighten_states['first']))
        print('straighten_states,second:'+str(straighten_states['second']))
        print('straighten_states,third:'+str(straighten_states['third']))
        print('bend_states,second:'+str(bend_states['second']))
        print('bend_states,third:'+str(bend_states['third']))

    if straighten_states['first'] :
        if straighten_states['second'] or bend_states['second'] or straighten_states['third'] or bend_states['third'] ==True:
            return False
        else:
           return  True
    else:
        print('straighten_states,first:'+str(straighten_states['first']))
        return False



def get_gripper_coordinates(data ):

    wrist=np.array(data[0])
    tips_first=np.array(data[4])
    tips_second=np.array(data[8])
    tips_third=np.array(data[12])

    # 计算两个端点与圆心构成的向量
    v1 = tips_second - wrist
    v2 = tips_third - wrist

    # 计算平面的法向量（两个向量的叉积）
    normal = np.cross(v1, v2)

    # 计算圆心在平面上的投影点
    proj_wrist = wrist - np.dot(wrist, normal) * normal / np.dot(normal, normal)

    # 计算两个端点的中点
    midpoint = (tips_second + tips_third) / 2

    # 将中点投影到平面上
    proj_midpoint = midpoint - np.dot(midpoint - proj_wrist, normal) * normal / np.dot(normal, normal)

    # 扇形弧线上的中点是圆心到投影中点的线段的中点
    arc_midpoint = (proj_wrist + proj_midpoint) / 2

    return tips_first, tips_first, arc_midpoint 
