# import numpy as np
# import cv2
#
#
# def brief_descriptor(image, keypoints, patch_size=32, n_pairs=256):
#     # 将图像转换为灰度图
#     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
#     # 初始化描述子矩阵，每一行代表一个特征点的描述子
#     descriptors = np.zeros((len(keypoints), n_pairs), dtype=np.uint8)
#
#     # 生成随机点对
#     points = np.random.randint(-patch_size // 2, patch_size // 2, (n_pairs, 2))
#
#     for i, keypoint in enumerate(keypoints):
#         x, y = int(keypoint.pt[0]), int(keypoint.pt[1])
#
#         # 对于每一对点，比较它们的灰度值
#         for j, (dx, dy) in enumerate(points):
#             if x + dx >= 0 and x + dx < gray_image.shape[1] and y + dy >= 0 and y + dy < gray_image.shape[0]:
#                 pixel1 = gray_image[y + dy, x + dx]
#                 pixel2 = gray_image[y, x]
#                 descriptors[i, j] = pixel1 > pixel2
#
#     return descriptors
#
#
# # 加载图像
# image = cv2.imread('pic2.png')
#
# # 检测特征点
# orb = cv2.ORB_create()
# keypoints, _ = orb.detectAndCompute(image, None)
#
# # 获取BRIEF描述子
# descriptors = brief_descriptor(image, keypoints)
#
# # 打印描述子
# print(descriptors)

import numpy as np
import cv2
import matplotlib.pyplot as plt


def brief_descriptor(image, keypoints, patch_size=32, n_pairs=256):
    # 将图像转换为灰度图
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 初始化描述子矩阵，每一行代表一个特征点的描述子
    descriptors = np.zeros((len(keypoints), n_pairs), dtype=np.uint8)

    # 生成随机点对
    points = np.random.randint(-patch_size // 2, patch_size // 2, (n_pairs, 2))

    for i, keypoint in enumerate(keypoints):
        x, y = int(keypoint.pt[0]), int(keypoint.pt[1])

        # 对于每一对点，比较它们的灰度值
        for j, (dx, dy) in enumerate(points):
            if x + dx >= 0 and x + dx < gray_image.shape[1] and y + dy >= 0 and y + dy < gray_image.shape[0]:
                pixel1 = gray_image[y + dy, x + dx]
                pixel2 = gray_image[y, x]
                descriptors[i, j] = pixel1 > pixel2

    return descriptors


# 加载两幅图像
image1 = cv2.imread('.\Images\pic_test_1.png')
image2 = cv2.imread('.\Images\pic_test_2.png')

# 检测特征点
orb = cv2.ORB_create()
keypoints1, descriptors1 = orb.detectAndCompute(image1, None)
keypoints2, descriptors2 = orb.detectAndCompute(image2, None)

# 使用BFMatcher进行匹配
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(descriptors1, descriptors2)

# 根据距离排序匹配结果
matches = sorted(matches, key=lambda x: x.distance)

# 绘制匹配结果
img_matches = np.empty((max(image1.shape[0], image2.shape[0]), image1.shape[1] + image2.shape[1], 3), dtype=np.uint8)
cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches[:50], img_matches,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# 展示匹配结果
plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()