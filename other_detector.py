import cv2
import matplotlib.pyplot as plt

# 读取图像
image = cv2.imread('./Images/pic_test_1.png', cv2.IMREAD_GRAYSCALE)

# 初始化SIFT检测器
sift = cv2.SIFT_create()

# 检测关键点和计算描述子
keypoints, descriptors = sift.detectAndCompute(image, None)

# 如果关键点数量超过250，选择响应值最高的250个点
if len(keypoints) > 200:
    # 根据响应值对关键点进行排序
    keypoints = sorted(keypoints, key=lambda x: x.response, reverse=True)
    # 选择前250个关键点
    keypoints = keypoints[:250]

# 绘制关键点
out = cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
kp=cv2.KeyPoint_convert(keypoints)
print(kp)
# 显示图像
cv2.imshow('Strong Keypoints', out)
cv2.waitKey(0)
cv2.destroyAllWindows()