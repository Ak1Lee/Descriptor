import cv2
import numpy as np

def fast_keypoint_detection(image, threshold=50, radius=3, circle_size=16):
    height, width = image.shape
    keypoints = []
    circle = np.zeros((circle_size, circle_size), dtype=np.uint8)
    center = (circle_size // 2, circle_size // 2)
    cv2.circle(circle, center, radius, 1, -1)

    for y in range(radius, height - radius):
        for x in range(radius, width - radius):
            center_pixel = image[y, x]
            y_start = y - radius
            y_end = y + radius + 1
            x_start = x - radius
            x_end = x + radius + 1
            if y_start < 0:
                y_start = 0
            if y_end > height:
                y_end = height
            if x_start < 0:
                x_start = 0
            if x_end > width:
                x_end = width
            circle_pixels = image[y_start:y_end, x_start:x_end]
            if circle_pixels.shape!= (circle_size, circle_size):
                temp = np.zeros((circle_size, circle_size), dtype=image.dtype)
                temp[:circle_pixels.shape[0], :circle_pixels.shape[1]] = circle_pixels
                circle_pixels = temp
            circle_masked = circle * circle_pixels
            diffs = np.abs(circle_masked - center_pixel)
            # 增加超过阈值的像素数量要求
            is_keypoint = np.sum(diffs > threshold) >= 15
            if is_keypoint:
                keypoints.append((x, y))
    return keypoints

# 读取图像并转换为灰度图
image = cv2.imread('Images/pic_test_1.png', 0)

# 进行 FAST 关键点检测
keypoints = fast_keypoint_detection(image)

# 在图像上绘制关键点
for keypoint in keypoints:
    cv2.circle(image, keypoint, 3, 255, -1)

# 显示结果图像
cv2.imshow('FAST Keypoints', image)
cv2.waitKey(0)
cv2.destroyAllWindows()