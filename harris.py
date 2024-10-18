import numpy as np
import cv2

def test():
    '''
    调用系统库函数进行测试
    '''
    img = cv2.imread('./Images/pic_test_1.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img = cv2.resize(img,dsize=(600,400))
    # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    #角点检测 第三个参数为角点检测的敏感度，其值必须介于3~31之间的奇数
    dst = cv2.cornerHarris(gray,3,3,0.04)
    print(dst.shape)  #(400, 600)
    img[dst>0.01*dst.max()] = [0,0,255]
    cv2.imshow('',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def harris_det(img, block_size=3, ksize=3, k=0.04, threshold=0.01, top_n=1000, suppression_radius=1):
    '''
    自己实现角点检测

    params:
        img:灰度图片
        block_size：自相关矩阵
        corner, top_corners
        k: trace_A的系数 lamda1*lambda2
        threshold:阈值
        top_n:返回前多少点
        suppression_radius：抑制半径
    return：

    '''
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


def harris_detect_other(img, ksize=3):
    '''
    实现角点检测

    params:
        img:灰度图片
        ksize：Sobel算子窗口大小

    return：
        corner：与源图像一样大小，角点处像素值设置为255
    '''
    k = 0.04  # 响应函数k
    threshold = 0.01  # 设定阈值
    WITH_NMS = True  # 是否非极大值抑制

    # 1、使用Sobel计算像素点x,y方向的梯度
    h, w = img.shape[:2]
    # Sobel函数求完导数后会有负值，还有会大于255的值。而原图像是uint8，即8位无符号数，所以Sobel建立的图像位数不够，会有截断。因此要使用16位有符号的数据类型，即cv2.CV_16S。
    grad = np.zeros((h, w, 2), dtype=np.float32)
    grad[:, :, 0] = cv2.Sobel(img, cv2.CV_16S, 1, 0, ksize=3)
    grad[:, :, 1] = cv2.Sobel(img, cv2.CV_16S, 0, 1, ksize=3)

    # 2、计算Ix^2,Iy^2,Ix*Iy
    m = np.zeros((h, w, 3), dtype=np.float32)
    m[:, :, 0] = grad[:, :, 0] ** 2
    m[:, :, 1] = grad[:, :, 1] ** 2
    m[:, :, 2] = grad[:, :, 0] * grad[:, :, 1]

    # 3、利用高斯函数对Ix^2,Iy^2,Ix*Iy进行滤波
    m[:, :, 0] = cv2.GaussianBlur(m[:, :, 0], ksize=(ksize, ksize), sigmaX=2)
    m[:, :, 1] = cv2.GaussianBlur(m[:, :, 1], ksize=(ksize, ksize), sigmaX=2)
    m[:, :, 2] = cv2.GaussianBlur(m[:, :, 2], ksize=(ksize, ksize), sigmaX=2)
    m = [np.array([[m[i, j, 0], m[i, j, 2]], [m[i, j, 2], m[i, j, 1]]]) for i in range(h) for j in range(w)]

    # 4、计算局部特征结果矩阵M的特征值和响应函数R(i,j)=det(M)-k(trace(M))^2  0.04<=k<=0.06
    D, T = list(map(np.linalg.det, m)), list(map(np.trace, m))
    R = np.array([d - k * t ** 2 for d, t in zip(D, T)])

    # 5、将计算出响应函数的值R进行非极大值抑制，滤除一些不是角点的点，同时要满足大于设定的阈值
    # 获取最大的R值
    R_max = np.max(R)
    # print(R_max)
    # print(np.min(R))
    R = R.reshape(h, w)
    corner = np.zeros_like(R, dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            if WITH_NMS:
                # 除了进行进行阈值检测 还对3x3邻域内非极大值进行抑制(导致角点很小，会看不清)
                if R[i, j] > R_max * threshold and R[i, j] == np.max(
                        R[max(0, i - 1):min(i + 2, h - 1), max(0, j - 1):min(j + 2, w - 1)]):
                    corner[i, j] = 255
            else:
                # 只进行阈值检测
                if R[i, j] > R_max * threshold:
                    corner[i, j] = 255
    # return corner
    # Get the coordinates of the top_n corners
    coordinates = np.argwhere(corner == 255)
    top_corners = coordinates if len(coordinates) <= 500 else coordinates[:500]

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


test()

image1 = cv2.imread('./Images/pic_test_1.png')
image2 = cv2.imread('./Images/pic_test_2.png')
# image1 = cv2.GaussianBlur(image1, (5, 5), 0)
# image2 = cv2.GaussianBlur(image2, (5, 5), 0)

# pic1
height, width, channel = image1.shape
print('image shape --> h:%d  w:%d  c:%d' % (height, width, channel))

gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), sigmaX=0)
# dst, corners_500 = harris_det(gray, top_n=500, suppression_radius=2)
dst_250, corners_250_1 = harris_det(gray, top_n=500, suppression_radius=2)
# Display the top 250 interest points
dst_250_img = image1.copy()
for y, x in corners_250_1:
    cv2.circle(dst_250_img, (x, y), 1, (0, 0, 255), -1)

# cv2.namedWindow('Top 250 Interest Points', cv2.WINDOW_NORMAL)

# cv2.imshow('Top 250 Interest Points', dst_250_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# img = cv2.imread('Image/img.png')
img = image1
# img = cv2.resize(img, dsize=(600, 400))
# 转换为灰度图像
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
keypoints, top_corners = harris_detect_other(gray)
[]
cv2.namedWindow('Top 250 Interest Points', cv2.WINDOW_NORMAL)

cv2.imshow('Top 250 Interest Points', dst_250_img)

dst_ = image1.copy()
for y, x in top_corners:
    cv2.circle(dst_, (x, y), 1, (0, 0, 255), -1)
cv2.namedWindow(' ', cv2.WINDOW_NORMAL)
cv2.imshow('', dst_)
cv2.waitKey(0)
cv2.destroyAllWindows()
