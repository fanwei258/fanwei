import os
import pydicom
import cv2
import skimage
import numpy as np
from matplotlib import pyplot as plt
from skimage.filters import gaussian
from skimage.segmentation import active_contour

skimage.coordinates = 'rc'


# 读取DICOM返回截取的血管图像
def extract_DICOM(DICOM):
    # dir_a = "./Images/"
    data = pydicom.dcmread(DICOM)
    rows = int(data.Rows)
    cols = int(data.Columns)
    imgArrary = data.pixel_array;
    imgNum = imgArrary.shape[0]
    print("图像数量：", imgNum)
    print("图像尺寸：", imgArrary.shape[1], " x ", imgArrary.shape[2])
    print("裁剪尺寸：570 x 570")
    for i in range(imgNum):
        img = data.pixel_array[i]
        img = img[:, :, 1]
        img_name = input_dir + str(i) + ".jpg"
        img_name_p = out_dir_p + str(i) + ".jpg"

        cropped = img[100:470, 340:710]
        # img_g = cv2.cvtColor(cropped,cv2.COLOR_BGR2GRAY)
        img_g = cropped;
        # img_p = cv2.bilateralFilter(img_g,18,90,100)
        img_p = cv2.medianBlur(img_g, 13)
        img_p = cv_filter2d(img_p)
        cv2.imwrite(img_name, cropped)
        cv2.imwrite(img_name_p, img_p)
    return imgNum


# 查找血管中心
def find_c(img):
    # 获取图像尺寸
    height = img.shape[0]
    width = img.shape[1]
    # 用于保存二值化图像
    result = np.zeros((height, width), np.uint8)
    # num统计参与计算中心点的像素点数量
    num = 0
    # X坐标累计
    x_sum = 0
    # Y坐标累计
    y_sum = 0
    # 二值化
    for i in range(height):
        for j in range(width):
            # 灰度值低于65变为0，其余为255
            if img[i, j] <= 35:
                gray = 0
            else:
                gray = 255
                num = num + 1
                x_sum = x_sum + j
                y_sum = y_sum + i
                # gray = int(img[i,j]*(255/50))
            result[i, j] = np.uint8(gray)
    x_mean = int(x_sum / num)
    y_mean = int(y_sum / num)
    print("图像大小：", height, width)
    print("预测中心点：", x_mean, y_mean)
    # cv2.circle(img,(x_mean,y_mean),5,(255,0,255),-1)
    # cv2.imshow("img",img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return x_mean, y_mean;


# 锐化
def cv_filter2d(img):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])

    dst = cv2.filter2D(img, -1, kernel)
    return dst


# 创建文件夹
def make_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def get_snake(img):
    # 中值滤波
    # dst = cv2.medianBlur(img ,13)
    # 锐化
    # dst = cv_filter2d(dst)
    # canny提取轮廓
    # dst = cv2.Canny(dst, 100, 200)
    dst = img;
    x_0, y_0 = find_c(dst)
    t = np.linspace(0, 2 * np.pi, 500)  # 参数t, [0,2π]
    x = x_0 + 195 * np.cos(t)
    y = y_0 + 195 * np.sin(t)
    # 构造初始Snake
    init = np.array([x, y]).T  # shape=(400, 2)
    # Snake模型迭代输出
    snake = active_contour(gaussian(dst, 3), snake=init, alpha=1.1, beta=2.0, gamma=0.01, w_line=0, w_edge=10)

    plt.figure(figsize=(5, 5))
    plt.imshow(dst, cmap="gray")
    plt.plot(snake[:, 0], snake[:, 1], '-w', lw=3)
    plt.xticks([]), plt.yticks([]), plt.axis("off")
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.savefig(out_dir + str(10) + '.jpg');
    plt.show()
    return snake


def img_add_edge(img, snake, path):
    plt.figure(figsize=(5, 5))
    plt.imshow(img, cmap="gray")
    plt.plot(snake[:, 0], snake[:, 1], '-w', lw=3)
    plt.xticks([]), plt.yticks([]), plt.axis("off")
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.savefig(path)
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    cv2.imwrite(path, gray)
    plt.close()
    # time.sleep(5)


# 创建输入输出临时文件
input_dir = './Images/'
make_dir(input_dir)
temp_dir = './Temp/'
make_dir(temp_dir)
out_dir = './Result/'
make_dir(out_dir)
out_dir_p = './Result_processed/'
make_dir(out_dir_p)

# 截取血管灰度图像
DICOM = "./IMG001.dcm"
# img_num = extract_DICOM(DICOM)

# 获取snake边缘
# img0 = cv2.imread("./test.jpg", 0)
# 第一张图像轮廓
# snake0 = get_snake(img0)

