import cv2
import numpy as np
import random
import my_orb_discriptor

def harris_det(img, block_size=3, ksize=3, k=0.04, threshold=0.01, top_n=1000, suppression_radius=1):
    h, w = img.shape[:2]
    # 1. Gaussian weighting
    gray = cv2.GaussianBlur(img, ksize=(ksize, ksize), sigmaX=2)

    # 2. Calculate gradients
    grad_x, grad_y = np.gradient(gray)
    Ix2 = grad_x ** 2
    Iy2 = grad_y ** 2
    Ixy = grad_x * grad_y

    # 3. Calculate covariance matrix elements
    window = np.ones((block_size, block_size), np.float32) / (block_size ** 2)
    sigma_xx = cv2.filter2D(Ix2, -1, window)
    sigma_xy = cv2.filter2D(Ixy, -1, window)
    sigma_yy = cv2.filter2D(Iy2, -1, window)

    # 4. Calculate Harris response function
    det_A = sigma_xx * sigma_yy - sigma_xy ** 2
    trace_A = sigma_xx + sigma_yy
    R = det_A - k * trace_A ** 2

    # Apply non-maximum suppression
    R_max = np.max(R)
    R = R / R_max
    corner = np.zeros_like(R, dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            if R[i, j] > threshold:
                max_val = R[i, j]
                for u in range(max(0, i - suppression_radius), min(i + suppression_radius + 1, h)):
                    for v in range(max(0, j - suppression_radius), min(j + suppression_radius + 1, w)):
                        if R[u, v] > max_val:
                            max_val = R[u, v]
                if R[i, j] == max_val:
                    corner[i, j] = 255

    # Get the coordinates of the top_n corners
    coordinates = np.argwhere(corner == 255)
    top_corners = coordinates if len(coordinates) <= top_n else coordinates[:top_n]

    return corner, top_corners


def harris_response(img, ksize=3, k=0.04):
    """
    Calculate the Harris response of an image.

    Parameters:
    - img: Input image.
    - ksize: Size of the Gaussian kernel.
    - k: Harris detector free parameter.

    Returns:
    - response: Harris response image.
    """
    h, w = img.shape[:2]
    # 1. Gaussian weighting
    gray = cv2.GaussianBlur(img, ksize=(ksize, ksize), sigmaX=2)

    # 2. Calculate gradients
    grad_x, grad_y = np.gradient(gray)
    Ix2 = grad_x ** 2
    Iy2 = grad_y ** 2
    Ixy = grad_x * grad_y

    # 3. Calculate covariance matrix elements
    window = np.ones((3, 3), np.float32) / 9
    sigma_xx = cv2.filter2D(Ix2, -1, window)
    sigma_xy = cv2.filter2D(Ixy, -1, window)
    sigma_yy = cv2.filter2D(Iy2, -1, window)

    # 4. Calculate Harris response function
    det_A = sigma_xx * sigma_yy - sigma_xy ** 2
    trace_A = sigma_xx + sigma_yy
    R = det_A - k * trace_A ** 2

    return R

def get_harris_points(response, threshold=0.01):
    """
    Get Harris points from the response image.

    Parameters:
    - response: Harris response image.
    - threshold: Threshold for Harris points.

    Returns:
    - response_t: Thresholded Harris response image.
    """
    corner_threshold = response.max() * threshold
    response_t = response * (response > corner_threshold)
    return response_t

def max_suppression(response, im_size, pad=1):
    """
    Apply non-maximum suppression to the Harris response image.

    Parameters:
    - response: Harris response image.
    - im_size: Size of the image.
    - pad: Suppression radius.

    Returns:
    - keypoints: Keypoints after non-maximum suppression.
    """
    h, w = im_size
    R_max = np.max(response)
    response = response / R_max
    corner = np.zeros_like(response, dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            if response[i, j] > 0:
                max_val = response[i, j]
                for u in range(max(0, i - pad), min(i + pad + 1, h)):
                    for v in range(max(0, j - pad), min(j + pad + 1, w)):
                        if response[u, v] > max_val:
                            max_val = response[u, v]
                if response[i, j] == max_val:
                    corner[i, j] = 255
    return corner

def harris_det(img, block_size=3, ksize=3, k=0.04, threshold=0.01, top_n=1000, suppression_radius=1):
    response = harris_response(img, ksize, k)
    response_t = get_harris_points(response, threshold)
    keypoints = max_suppression(response_t, img.shape[:2], suppression_radius)

    # Get the coordinates of the top_n corners
    coordinates = np.argwhere(keypoints == 255)
    top_corners = coordinates if len(coordinates) <= top_n else coordinates[:top_n]

    return keypoints, top_corners

def compute_descriptors(img, keypoints, num_bins=8, dsize=(16, 16)):
    descriptors = []
    for kp in keypoints:
        # 计算关键点描述子所需的图像区域
        x, y = int(kp[0]), int(kp[1])
        s = int(kp.size)
        sigma = s / 2

        # 计算旋转后的坐标
        mask = np.zeros((dsize[1], dsize[0]), dtype=np.float32)
        angle = kp.angle
        cx, cy = dsize[0] // 2, dsize[1] // 2
        x, y = x - cy, y - cx
        for i in range(dsize[1]):
            for j in range(dsize[0]):
                if angle == 0:
                    new_x, new_y = j, i
                else:
                    new_x = j * np.cos(angle) + i * np.sin(angle)
                    new_y = -j * np.sin(angle) + i * np.cos(angle)
                new_x += cx
                new_y += cy
                if 0 <= new_x < img.shape[1] and 0 <= new_y < img.shape[0]:
                    mask[i, j] = img[new_y, new_x]

        # 计算梯度直方图
        hist = np.zeros((num_bins, num_bins), dtype=np.float32)
        for i in range(dsize[1]):
            for j in range(dsize[0]):
                gradient = cv2.Sobel(mask, cv2.CV_32F, 1, 0, ksize=5)[j, i]
                gradient = np.sqrt(gradient ** 2)
                orientation = cv2.Sobel(mask, cv2.CV_32F, 0, 1, ksize=5)[j, i] / gradient
                orientation = np.arctan2(orientation, 1) * 180 / np.pi
                bin_x = int((num_bins - 1) * (j / dsize[0]))
                bin_y = int((num_bins - 1) * (i / dsize[1]))
                if 0 <= bin_x < num_bins and 0 <= bin_y < num_bins:
                    hist[bin_y, bin_x] += gradient

        # 归一化直方图
        hist = hist.ravel()
        hist = hist / np.linalg.norm(hist)

        descriptors.append(hist)

    return np.array(descriptors)
def compute_sift_descriptors(img, corners, num_bins=8, dsize=(16, 16)):
    descriptors = []
    for corner in corners:
        x, y = int(corner[0]), int(corner[1])
        sigma = 1.6  # 通常SIFT算法中使用的高斯模糊系数

        # 提取关键点周围的邻域
        patch = img[max(0, y-dsize[1]//2):min(img.shape[0], y+dsize[1]//2),
                    max(0, x-dsize[0]//2):min(img.shape[1], x+dsize[0]//2)]
        if patch.shape[0] < dsize[1] or patch.shape[1] < dsize[0]:
            continue  # 如果邻域小于所需的尺寸，则跳过这个关键点

        # 对邻域进行高斯模糊
        patch = cv2.GaussianBlur(patch, (0, 0), sigma)

        # 初始化方向直方图
        hist = np.zeros((num_bins, num_bins), dtype=np.float32)

        # 计算每个4x4子区域的梯度直方图
        for i in range(0, dsize[1], 4):
            for j in range(0, dsize[0], 4):
                subpatch = patch[i:i+4, j:j+4]
                gradients = cv2.Sobel(subpatch, cv2.CV_32F, 1, 0, ksize=3)
                orientations = cv2.Sobel(subpatch, cv2.CV_32F, 0, 1, ksize=3)
                angle = np.arctan2(orientations, gradients) * 180 / np.pi

                # 分配梯度到8个方向的直方图
                for m in range(4):
                    for n in range(4):
                        mag = np.sqrt(gradients[m, n]**2 + orientations[m, n]**2)
                        angle_idx = int((angle[m, n] / 360) * num_bins)
                        if angle_idx == num_bins:
                            angle_idx = 0
                        hist[angle_idx, i//4 * 4 + n, j//4 * 4 + m] += mag

        # 归一化直方图
        hist = hist.ravel()
        hist = hist / np.linalg.norm(hist)

        descriptors.append(hist)

    return np.array(descriptors)


def compute_orientation_histogram(img, keypoint, num_bins=36, window_size=16):
    # 计算邻域窗口的起始和结束坐标
    x1 = int(keypoint[0] - window_size / 2)
    y1 = int(keypoint[1] - window_size / 2)
    x2 = int(keypoint[0] + window_size / 2)
    y2 = int(keypoint[1] + window_size / 2)

    # 提取邻域窗口
    window = img[y1:y2, x1:x2]
    gray = cv2.GaussianBlur(img, ksize=(3, 3), sigmaX=2)
    # 计算梯度
    grad_x, grad_y = np.gradient(gray)

    # 计算梯度幅度和方向
    magnitude = cv2.magnitude(grad_x, grad_y)
    orientation = cv2.phase(grad_x, grad_y, angleInDegrees=True)

    # 初始化方向直方图
    hist = np.zeros(num_bins, dtype=np.float32)

    # 计算高斯权重
    sigma = 1.5 * window_size / 2
    gx = np.arange(window_size) - window_size / 2
    gy = np.arange(window_size) - window_size / 2
    G = np.exp(-(gx[:, None] ** 2 + gy[None, :] ** 2) / (2 * sigma ** 2))

    # 构建方向直方图
    for i in range(window_size):
        for j in range(window_size):
            angle_bin = int(num_bins * (orientation[x1 + i, y1 + j] / 360))
            if angle_bin == num_bins:
                angle_bin = 0
            hist[angle_bin] += magnitude[x1 + i, y1 + j] * G[i, j]

    # 归一化直方图
    hist = hist / (np.sum(hist)+0.0001)

    return hist

def gaussian_noise(image, sigma=2):
    """高斯噪声"""
    mean = 0
    gaussian = np.random.normal(mean, sigma, image.shape)
    noisy = image + gaussian
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy

def generate_random_pairs(num_pairs, patch_size, stride=8):
    """生成随机点对"""
    pairs = []
    for _ in range(num_pairs):
        x1 = np.random.randint(0, patch_size)
        y1 = np.random.randint(0, patch_size)
        x2 = np.random.randint(0, patch_size)
        y2 = np.random.randint(0, patch_size)
        while (x1 - x2) ** 2 + (y1 - y2) ** 2 < stride ** 2:  # 确保点对之间的距离足够远
            x2 = np.random.randint(0, patch_size)
            y2 = np.random.randint(0, patch_size)
        pairs.append((x1, y1, x2, y2))
    return np.array(pairs)


def brief_descriptor(image, keypoints, num_pairs=256, patch_size=31, stride=8):
    """计算BRIEF描述子"""
    descriptors = []
    for keypoint in keypoints:
        x, y = int(keypoint[1]), int(keypoint[0])
        pairs = generate_random_pairs(num_pairs, patch_size)

        # 计算关键点邻域窗口的边界
        x_min = max(0, x - patch_size // 2)
        x_max = min(image.shape[1], x + patch_size // 2)
        y_min = max(0, y - patch_size // 2)
        y_max = min(image.shape[0], y + patch_size // 2)

        descriptor = []
        for x1, y1, x2, y2 in pairs:
            # 调整点对坐标以适应关键点邻域窗口
            x1 += x - patch_size // 2
            y1 += y - patch_size // 2
            x2 += x - patch_size // 2
            y2 += y - patch_size // 2

            # 边界检查
            x1 = max(x_min, min(x1, x_max))
            y1 = max(y_min, min(y1, y_max))
            x2 = max(x_min, min(x2, x_max))
            y2 = max(y_min, min(y2, y_max))

            # 获取像素值
            v1 = image[y1, x1]
            v2 = image[y2, x2]

            # 比较像素值并生成描述子
            if v1 > v2:
                descriptor.append(1)
            else:
                descriptor.append(0)

        # 将描述子转换为numpy数组
        descriptors.append(np.array(descriptor))

    return np.array(descriptors)


if __name__ == "__main__":
    image1 = cv2.imread('./Images/pic_test_1.png')
    image2 = cv2.imread('./Images/pic_test_2.png')
    image1 = cv2.GaussianBlur(image1, (5, 5), 0)
    image2 = cv2.GaussianBlur(image2, (5, 5), 0)

    # pic1
    height, width, channel = image1.shape
    print('image shape --> h:%d  w:%d  c:%d' % (height, width, channel))
    # cv2.imshow('Original Image', image1)
    # cv2.waitKey(200)
    # cv2.destroyAllWindows()
    gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), sigmaX=0)
    # dst, corners_500 = harris_det(gray, top_n=500, suppression_radius=2)
    dst_250, corners_250_1 = harris_det(gray, top_n=150, suppression_radius=2)
    # Display the top 250 interest points
    dst_250_img = image1.copy()
    for y, x in corners_250_1:
        cv2.circle(dst_250_img, (x, y), 5, (0, 0, 255), -1)
    cv2.imshow('Top 250 Interest Points', dst_250_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    Q, Theta_Set = my_orb_discriptor.get_orb_theta_pos(gray,corners_250_1)
    arr1 = my_orb_discriptor.get_orb_descriptor(gray,corners_250_1,Theta_Set)



    #
    # pic2
    #
    height, width, channel = image2.shape
    print('image shape --> h:%d  w:%d  c:%d' % (height, width, channel))
    # cv2.imshow('Original Image', image1)
    # cv2.waitKey(200)
    # cv2.destroyAllWindows()
    gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    # dst, corners_500 = harris_det(gray, top_n=500, suppression_radius=2)
    dst_250, corners_250_2 = harris_det(gray, top_n=150, suppression_radius=2)
    # Display the top 250 interest points
    dst_250_img = image2.copy()
    for y, x in corners_250_2:
        cv2.circle(dst_250_img, (x, y), 5, (0, 0, 255), -1)
    cv2.imshow('Top 250 Interest Points', dst_250_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    Q, Theta_Set = my_orb_discriptor.get_orb_theta_pos(gray,corners_250_2)
    arr2 = my_orb_discriptor.get_orb_descriptor(gray,corners_250_2,Theta_Set)

    pos_pair = my_orb_discriptor.get_pos_match(arr1,arr2,corners_250_1,corners_250_2)

    # 创建一个新的画布，大小足以容纳两张图片
    height = max(image1.shape[0], image2.shape[0])
    width = image1.shape[1] + image2.shape[1]

    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    # 将第一张图片放在画布的左侧
    canvas[:image1.shape[0], :image1.shape[1]] = image1

    # 将第二张图片放在画布的右侧
    canvas[:image2.shape[0], image1.shape[1]:] = image2

    #     use sift dectetor
    # 初始化SIFT检测器
    sift = cv2.SIFT_create()

    # 检测关键点和计算描述子
    keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
    keypoints2, descriptors1 = sift.detectAndCompute(image2, None)
    # 如果关键点数量超过250，选择响应值最高的250个点
    if len(keypoints1) > 500:
        # 根据响应值对关键点进行排序
        keypoints1 = sorted(keypoints1, key=lambda x: x.response, reverse=True)
        # 选择前250个关键点
        keypoints1 = keypoints1[:500]
    if len(keypoints2) > 500:
        # 根据响应值对关键点进行排序
        keypoints2 = sorted(keypoints2, key=lambda x: x.response, reverse=True)
        # 选择前250个关键点
        keypoints2 = keypoints2[:500]

    kp1 = np.floor(cv2.KeyPoint_convert(keypoints1))
    kp2 = np.floor(cv2.KeyPoint_convert(keypoints2))
    kp1 = kp1[:, [1,0]]
    kp2 = kp2[:, [1,0]]

    # kp1 = np.array([[83,332]])
    # kp2 = np.array([[83,436]])
    print(kp1)


    Q, Theta_Set = my_orb_discriptor.get_orb_theta_pos(gray,kp1)
    arr1 = my_orb_discriptor.get_orb_descriptor(gray,kp1,Theta_Set)

    Q, Theta_Set = my_orb_discriptor.get_orb_theta_pos(gray,kp2)
    arr2 = my_orb_discriptor.get_orb_descriptor(gray,kp2,Theta_Set)

    pos_pair = my_orb_discriptor.get_pos_match(arr1, arr2, kp1, kp2)
    print(pos_pair)


    # 在两张图片的对应点之间画线
    # line_colors = [(0, 255, 0), (0, 0, 255)]  # 线的颜色，例如：绿色和蓝色
    i = 0
    for pair in pos_pair:
        if(pair[0] == -1):
            continue
        start_point = (pair[1], pair[0])
        end_point = (pair[3] + image1.shape[1], pair[2])
        random_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        cv2.line(canvas, start_point, end_point,random_color,1)  # 绿色线段
    # 显示最终的画布
    cv2.imshow('Connected Points', canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

