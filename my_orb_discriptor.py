import numpy as np
import cv2

# orb取点模板
bit_pattern_31 = [
    8,-3, 9,5,
    4,2, 7,-12,
    -11,9, -8,2,
    7,-12, 12,-13,
    2,-13, 2,12,
    1,-7, 1,6,
    -2,-10, -2,-4,
    -13,-13, -11,-8,
    -13,-3, -12,-9,
    10,4, 11,9,
    -13,-8, -8,-9,
    -11,7, -9,12,
    7,7, 12,6,
    -4,-5, -3,0,
    -13,2, -12,-3,
    -9,0, -7,5,
    12,-6, 12,-1,
    -3,6, -2,12,
    -6,-13, -4,-8,
    11,-13, 12,-8,
    4,7, 5,1,
    5,-3, 10,-3,
    3,-7, 6,12,
    -8,-7, -6,-2,
    -2,11, -1,-10,
    -13,12, -8,10,
    -7,3, -5,-3,
    -4,2, -3,7,
    -10,-12, -6,11,
    5,-12, 6,-7,
    5,-6, 7,-1,
    1,0, 4,-5,
    9,11, 11,-13,
    4,7, 4,12,
    2,-1, 4,4,
    -4,-12, -2,7,
    -8,-5, -7,-10,
    4,11, 9,12,
    0,-8, 1,-13,
    -13,-2, -8,2,
    -3,-2, -2,3,
    -6,9, -4,-9,
    8,12, 10,7,
    0,9, 1,3,
    7,-5, 11,-10,
    -13,-6, -11,0,
    10,7, 12,1,
    -6,-3, -6,12,
    10,-9, 12,-4,
    -13,8, -8,-12,
    -13,0, -8,-4,
    3,3, 7,8,
    5,7, 10,-7,
    -1,7, 1,-12,
    3,-10, 5,6,
    2,-4, 3,-10,
    -13,0, -13,5,
    -13,-7, -12,12,
    -13,3, -11,8,
    -7,12, -4,7,
    6,-10, 12,8,
    -9,-1, -7,-6,
    -2,-5, 0,12,
    -12,5, -7,5,
    3,-10, 8,-13,
    -7,-7, -4,5,
    -3,-2, -1,-7,
    2,9, 5,-11,
    -11,-13, -5,-13,
    -1,6, 0,-1,
    5,-3, 5,2,
    -4,-13, -4,12,
    -9,-6, -9,6,
    -12,-10, -8,-4,
    10,2, 12,-3,
    7,12, 12,12,
    -7,-13, -6,5,
    -4,9, -3,4,
    7,-1, 12,2,
    -7,6, -5,1,
    -13,11, -12,5,
    -3,7, -2,-6,
    7,-8, 12,-7,
    -13,-7, -11,-12,
    1,-3, 12,12,
    2,-6, 3,0,
    -4,3, -2,-13,
    -1,-13, 1,9,
    7,1, 8,-6,
    1,-1, 3,12,
    9,1, 12,6,
    -1,-9, -1,3,
    -13,-13, -10,5,
    7,7, 10,12,
    12,-5, 12,9,
    6,3, 7,11,
    5,-13, 6,10,
    2,-12, 2,3,
    3,8, 4,-6,
    2,6, 12,-13,
    9,-12, 10,3,
    -8,4, -7,9,
    -11,12, -4,-6,
    1,12, 2,-8,
    6,-9, 7,-4,
    2,3, 3,-2,
    6,3, 11,0,
    3,-3, 8,-8,
    7,8, 9,3,
    -11,-5, -6,-4,
    -10,11, -5,10,
    -5,-8, -3,12,
    -10,5, -9,0,
    8,-1, 12,-6,
    4,-6, 6,-11,
    -10,12, -8,7,
    4,-2, 6,7,
    -2,0, -2,12,
    -5,-8, -5,2,
    7,-6, 10,12,
    -9,-13, -8,-8,
    -5,-13, -5,-2,
    8,-8, 9,-13,
    -9,-11, -9,0,
    1,-8, 1,-2,
    7,-4, 9,1,
    -2,1, -1,-4,
    11,-6, 12,-11,
    -12,-9, -6,4,
    3,7, 7,12,
    5,5, 10,8,
    0,-4, 2,8,
    -9,12, -5,-13,
    0,7, 2,12,
    -1,2, 1,7,
    5,11, 7,-9,
    3,5, 6,-8,
    -13,-4, -8,9,
    -5,9, -3,-3,
    -4,-7, -3,-12,
    6,5, 8,0,
    -7,6, -6,12,
    -13,6, -5,-2,
    1,-10, 3,10,
    4,1, 8,-4,
    -2,-2, 2,-13,
    2,-12, 12,12,
    -2,-13, 0,-6,
    4,1, 9,3,
    -6,-10, -3,-5,
    -3,-13, -1,1,
    7,5, 12,-11,
    4,-2, 5,-7,
    -13,9, -9,-5,
    7,1, 8,6,
    7,-8, 7,6,
    -7,-4, -7,1,
    -8,11, -7,-8,
    -13,6, -12,-8,
    2,4, 3,9,
    10,-5, 12,3,
    -6,-5, -6,7,
    8,-3, 9,-8,
    2,-12, 2,8,
    -11,-2, -10,3,
    -12,-13, -7,-9,
    -11,0, -10,-5,
    5,-3, 11,8,
    -2,-13, -1,12,
    -1,-8, 0,9,
    -13,-11, -12,-5,
    -10,-2, -10,11,
    -3,9, -2,-13,
    2,-3, 3,2,
    -9,-13, -4,0,
    -4,6, -3,-10,
    -4,12, -2,-7,
    -6,-11, -4,9,
    6,-3, 6,11,
    -13,11, -5,5,
    11,11, 12,6,
    7,-5, 12,-2,
    -1,12, 0,7,
    -4,-8, -3,-2,
    -7,1, -6,7,
    -13,-12, -8,-13,
    -7,-2, -6,-8,
    -8,5, -6,-9,
    -5,-1, -4,5,
    -13,7, -8,10,
    1,5, 5,-13,
    1,0, 10,-13,
    9,12, 10,-1,
    5,-8, 10,-9,
    -1,11, 1,-13,
    -9,-3, -6,2,
    -1,-10, 1,12,
    -13,1, -8,-10,
    8,-11, 10,-6,
    2,-13, 3,-6,
    7,-13, 12,-9,
    -10,-10, -5,-7,
    -10,-8, -8,-13,
    4,-6, 8,5,
    3,12, 8,-13,
    -4,2, -3,-3,
    5,-13, 10,-12,
    4,-13, 5,-1,
    -9,9, -4,3,
    0,3, 3,-9,
    -12,1, -6,1,
    3,2, 4,-8,
    -10,-10, -10,9,
    8,-13, 12,12,
    -8,-12, -6,-5,
    2,2, 3,7,
    10,6, 11,-8,
    6,8, 8,-12,
    -7,10, -6,5,
    -3,-9, -3,9,
    -1,-13, -1,5,
    -3,-7, -3,4,
    -8,-2, -8,3,
    4,2, 12,12,
    2,-5, 3,11,
    6,-9, 11,-13,
    3,-1, 7,12,
    11,-1, 12,4,
    -3,0, -3,6,
    4,-11, 4,12,
    2,-4, 2,1,
    -10,-6, -8,1,
    -13,7, -11,1,
    -13,12, -11,-13,
    6,0, 11,-13,
    0,-1, 1,4,
    -13,3, -9,-2,
    -9,8, -6,-3,
    -13,-6, -8,-2,
    5,-9, 8,10,
    2,7, 3,-9,
    -1,-6, -1,-1,
    9,5, 11,-2,
    11,-3, 12,-8,
    3,0, 3,5,
    -1,4, 0,10,
    3,-6, 4,5,
    -13,0, -10,5,
    5,8, 12,11,
    8,9, 9,-6,
    7,-4, 8,-12,
    -10,4, -10,9,
    7,3, 12,4,
    9,-7, 10,-2,
    7,0, 12,-2,
    -1,-6, 0,-11
]

bit_pattern_31 = np.array(bit_pattern_31).reshape(256, 4)
# 取得角度，质点
def get_orb_theta_pos(gray_img,corners,window_w = 30):
    theta_set = np.zeros(corners.shape[0])
    Q_set = np.zeros((corners.shape[0],2))
    num = 0

    for corner in corners:
        y_max = int(min(corner[0]+(window_w/2), gray_img.shape[0]))
        x_max = int(min(corner[1]+(window_w/2), gray_img.shape[1]))

        y_min = int(max(corner[0]-(window_w/2),0))
        x_min = int(max(corner[1]-(window_w/2),0))

        m00 = 0
        m01 = 0
        m10 = 0

        for y in range(0,y_max - y_min):
            for x in range(0,x_max - x_min):
                if((y+y_min - corner[0])**2 + (x+x_min-corner[1])**2) <= ((window_w/2)**2):
                    m00 += gray_img[y+y_min][x+x_min]
                    m10 += (y+y_min - corner[0]) * gray_img[y+y_min][x+x_min]
                    m01 += (x+x_min - corner[1]) * gray_img[y+y_min][x+x_min]

        q = np.array([round(corner[0] + m10/m00),round(corner[1] + m01/m00)])

        # 计算旋转角度
        theta = np.arctan2(m01, m10)
        theta_set[num] = theta
        Q_set[num] = q
        num+=1
        # 将角度转换为度数
        theta_degrees = np.degrees(theta)
        # #
        # print(f"旋转角度（弧度）: {theta}")
        # print(f"旋转角度（度数）: {theta_degrees}")



    return Q_set,theta_set

def is_in_range(xmax, xmin, ymax, ymin, arr2d):
    for point in arr2d:
        if point[1] > xmax or point[1] < xmin or point[0] > ymax or point[0] < ymin:
            return False
    return True
def hamming_distance(a, b):
    if len(a) != len(b):
        raise ValueError("输入数组必须具有相同的长度。")

    # 使用 XOR 操作比较两个数组，然后计算结果中非零元素的数量
    distance = np.count_nonzero(a ^ b)
    return distance
def get_orb_descriptor(gray_img, corners,theta_set):
    bit_pattern_length = bit_pattern_31.shape[0]
    arr = np.zeros((corners.shape[0],bit_pattern_length), dtype=int)
    for idx in range(corners.shape[0]):
        corner = corners[idx]
        theta = theta_set[idx]
        y_max = int(gray_img.shape[0]) - 1
        x_max = int(gray_img.shape[1]) - 1
        y_min = 0
        x_min = 0
        x = corner[1]
        y = corner[0]

        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        for i in range(bit_pattern_length):
            x1 = round((cos_theta * bit_pattern_31[i][1] - sin_theta * bit_pattern_31[i][0]) + x)
            y1 = round((sin_theta * bit_pattern_31[i][1] + cos_theta * bit_pattern_31[i][0]) + y)
            x2 = round((cos_theta * bit_pattern_31[i][3] - sin_theta * bit_pattern_31[i][2]) + x)
            y2 = round((sin_theta * bit_pattern_31[i][3] + cos_theta * bit_pattern_31[i][2]) + y)

            if is_in_range(x_max, x_min, y_max, y_min, [[y1, x1], [y2, x2]]):
                if gray_img[y1, x1] > gray_img[y2, x2]:
                    arr[idx][i] = 1

    return arr


def get_pos_match(descriptor1,descriptor2,corners1,corners2):
    size = (corners1.shape[0])
    return_pair = np.zeros((size,4),int)
    # return_pair = []

    for i in range(size):
        haming_min = np.inf
        for j in range(corners2.shape[0]):
            tmp = hamming_distance(descriptor1[i],descriptor2[j])
            if(tmp < haming_min):
                haming_min = tmp
                cor2 = corners2[j]

        print(haming_min)
        if(haming_min>60):
            continue
        return_pair = np.vstack((return_pair, np.array([corners1[i][0], corners1[i][1], cor2[0], cor2[1]])))
        return_pair = np.array(return_pair)
    return return_pair