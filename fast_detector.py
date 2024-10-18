import numpy as np
import cv2

check_circle = ((-1, 3), (0, 3), (1, 3), (2, 2),
                (3, 1), (3, 0), (3, -1), (2, -2),
                (1, -3), (0, -3), (-1, -3), (-2, -2),
                (-3, -1), (-3, 0), (-3, 1), (-2, 2))


def fast(img, t=0.2, n=9):
    print("开始计算fast")
    base_cut = img[3:img.shape[0] - 3, 3:img.shape[1] - 3]
    T_cut = base_cut * t
    # T_cut = np.empty(base_cut.shape)
    # T_cut.fill(20)
    checks = []
    for i in range(16):
        tmp = img[3 + check_circle[i][0]:img.shape[0] - 3 + check_circle[i][0],
              3 + check_circle[i][1]:img.shape[1] - 3 + check_circle[i][1]]
        check_array = base_cut - tmp
        checks.append(check_array)

    # 首先对全图中每个点检查1,5,9,13号位进行粗筛
    def rough_check():
        print("开始粗筛")
        # 我们认为我们关注的物体一般不会处于图像的边缘，所以我们将原图像与平移图像都切除边缘相减，简化处理的同时不影响效果
        res_array = np.zeros(base_cut.shape)
        for i in [1, 5, 9, 13]:
            res_array[np.abs(checks[i]) > T_cut] += 1
        return res_array

    # 选取特征点,
    def slight_check():
        pro_array = rough_check()
        print("开始细筛")
        res_array = np.zeros(base_cut.shape)
        for x in range(base_cut.shape[0]):
            for y in range(base_cut.shape[1]):
                # if pro_array[x, y] < 3:
                #     continue

                # goushi
                f = i = -1
                for j in range(16):
                    check = abs(checks[j][x, y])
                    Tar = T_cut[x, y]
                    if check > Tar:
                        if j - i - 1 >= n:
                            # res_array[x,y]=1 # 这种情况f+15-i>=12肯定也是满足的
                            break
                        i = j
                        if f == -1:
                            f = j
                if f + 15 - i >= n:
                    res_array[x, y] = 1

        return res_array

    fast_keys = slight_check()

    # 进行非最大值抑制
    def max_restrain(key_array, r=2):
        print("开始非最大值抑制")
        key_array = np.pad(key_array, ((3, 3), (3, 3)))
        key_point = np.where(key_array)
        s_array = np.zeros(base_cut.shape)
        masked_array = key_array.copy()
        for i in range(16):
            s_array += abs(checks[i])
        s_array = np.pad(s_array, ((3, 3), (3, 3)))
        for i in range(key_point[0].shape[0]):
            x = key_point[0][i]
            y = key_point[1][i]
            argmax = np.argmax(s_array[x - r:x + r + 1, y - r:y + r + 1])
            if argmax != (2 * r + 1) * r + r:
                masked_array[x, y] = 0
        return masked_array

    masked_array = max_restrain(fast_keys)

    return masked_array


def cal_5_5_sum_dif_abs(img, x, y):
    center_val = img[y][x]
    sum = 0
    for i in range(y - 2, y + 2):
        for j in range(x - 2, x + 2):
            sum += abs(img[i][j] - center_val)
    return sum


def my_fast(img, t=0.1, n=9):
    y_max = img.shape[0]
    x_max = img.shape[1]
    y_start = 15
    y_end = img.shape[0] - 15 - 1
    x_start = 15
    x_end = img.shape[1] - 15 - 1
    return_array = []
    score_map = np.zeros(img.shape)
    res_map = np.zeros(img.shape)
    for y in range(y_start, y_end):
        for x in range(x_start, x_end):
            # if (y == 155) & (x == 14):
            #     print(1)
            center_val = img[y][x]
            standard = max(15, center_val * t)
            num = 0
            is_find = False
            for i in range(16):
                val_tmp = abs(img[y + check_circle[i][0]][x + check_circle[i][1]] - center_val)
                num = 0
                if (val_tmp > standard):
                    num += 1
                    j = i
                    loop = 0
                    while 1:
                        j = (j + 1) % 16
                        val_tmp = abs(img[y + check_circle[j][0]][x + check_circle[j][1]] - center_val)
                        if (val_tmp > standard):
                            num += 1
                        else:
                            break
                        if num >= n:
                            is_find = True
                            break
                if (is_find):
                    score_map[y][x] = cal_5_5_sum_dif_abs(img, x, y)
                    res_map[y][x] = 1
                    break
    # 非最大值抑制
    for y in range(y_start, y_end):
        for x in range(x_start, x_end):
            if res_map[y][x] == 1:
                for dy in [-2, -1, 0, 1, 2]:
                    for dx in [-2, -1, 0, 1, 2]:
                        if score_map[y + dy][x + dx] > score_map[y][x]:
                            res_map[y][x] = 0
                            score_map[y][x] = 0
                            break
                    else:
                        continue
                    break
    for y in range(y_start, y_end):
        for x in range(x_start, x_end):
            if (res_map[y][x] != 0):
                return_array.append((y, x))
    return_array = np.array(return_array)

    # 提取分数和坐标
    # scores = score_map[y_start:y_end + 1, x_start:x_end + 1].flatten()

    scores = score_map[score_map!= 0].flatten()
    coords = np.argwhere(res_map == 1)

    # 合并坐标和分数
    data = np.column_stack((coords, scores))

    # 根据分数排序
    sorted_data = data[data[:, -1].argsort()][::-1]

    # 选择分数最高的250个点
    top_250 = sorted_data[:150]

    # 转换为最终的返回数组
    return_array = top_250[:, :2]

    return return_array


    # return return_array


image1 = cv2.imread('./Images/room1.png')
image2 = cv2.imread('./Images/room2.png')
image_miku_r = cv2.imread('./Images/miku2.png')



gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray = gray.astype(np.int32)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
gray2 = gray2.astype(np.int32)

#
image_show = image1.copy()
pos = my_fast(gray)
for p in pos:
    cv2.circle(image_show, (int(p[1]), int(p[0])), 2, (0, 0, 255), -1)
# cv2.namedWindow('1mk', cv2.WINDOW_NORMAL)
cv2.imshow('1mk', image_show)
cv2.waitKey(0)



# 222222222222
image_show2 = image2.copy()
pos2 = my_fast(gray2)
for p in pos2:
    cv2.circle(image_show2, (int(p[1]), int(p[0])), 2, (0, 0, 255), -1)
# cv2.namedWindow('2mk', cv2.WINDOW_NORMAL)
# cv2.imshow('2mk', image_show2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

import my_orb_discriptor

# 11111111111
Q,thetaSet = my_orb_discriptor.get_orb_theta_pos(gray,pos,window_w = 30)
for i in range(Q.shape[0]):
    cv2.line(image_show, (int(pos[i][1]), int(pos[i][0])), (int(Q[i][1]),int(Q[i][0])), (0, 255, 0), 1)
    # cv2.circle(image_show, (int(p[1]), int(p[0])), 2, (0, 0, 255), -1)
# cv2.namedWindow(' ', cv2.WINDOW_NORMAL)
# cv2.imshow('1', image_show)

# discriptor = my_orb_discriptor.get_orb_descriptor(gray,pos,thetaSet)
# 22222222
Q,thetaSet2 = my_orb_discriptor.get_orb_theta_pos(gray2,pos2,window_w = 30)
for i in range(Q.shape[0]):
    cv2.line(image_show2, (int(pos2[i][1]), int(pos2[i][0])), (int(Q[i][1]),int(Q[i][0])), (0, 255, 0), 1)
cv2.namedWindow(' ', cv2.WINDOW_NORMAL)

cv2.imshow('2', image_show2)

# cv2.waitKey(0)
# cv2.destroyAllWindows()


# pipei
discriptor1 = my_orb_discriptor.get_orb_descriptor(gray,pos,thetaSet)
discriptor2 = my_orb_discriptor.get_orb_descriptor(gray2,pos2,thetaSet2)

pos_pair = my_orb_discriptor.get_pos_match(discriptor1,discriptor2,pos,pos2)

# 创建一个新的画布，大小足以容纳两张图片
height = max(image1.shape[0], image2.shape[0])
width = image1.shape[1] + image2.shape[1]

canvas = np.zeros((height, width, 3), dtype=np.uint8)
# 将第一张图片放在画布的左侧
canvas[:image1.shape[0], :image1.shape[1]] = image1
# 将第二张图片放在画布的右侧
canvas[:image2.shape[0], image1.shape[1]:] = image2
# 在两张图片的对应点之间画线
# line_colors = [(0, 255, 0), (0, 0, 255)]  # 线的颜色，例如：绿色和蓝色
i = 0
import random
for pair in pos_pair:
    if (pair[0] == -1):
        continue
    start_point = (int(pair[1]), int(pair[0]))
    end_point = (int(pair[3] + image1.shape[1]), int(pair[2]))


    random_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    random_color = (0,0,255)
    cv2.circle(canvas, start_point, 2, random_color, -1)
    cv2.circle(canvas, end_point, 2, random_color, -1)

    cv2.line(canvas, start_point, end_point, random_color, 1)
# 显示最终的画布
cv2.namedWindow('Connected Points', cv2.WINDOW_NORMAL)
cv2.imshow('Connected Points', canvas)
cv2.moveWindow('Connected Points', 200, 200)
cv2.waitKey(0)
cv2.destroyAllWindows()