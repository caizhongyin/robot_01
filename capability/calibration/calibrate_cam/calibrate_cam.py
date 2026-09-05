import cv2
import numpy as np
import glob

def get_cam_dist(cal_img_path):
    images = glob.glob(cal_img_path)
    # 设置寻找亚像素角点的参数，采用的停止准则是最大循环次数30和最大误差容限0.001
    criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)

    # 获取标定板角点的位置
    objp = np.zeros((8 * 11, 3), np.float32)
    objp[:, :2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2)  # 将世界坐标系建在标定板上，所有点的Z坐标全部为0，所以只需要赋值x和y

    obj_points = []  # 存储3D点
    img_points = []  # 存储2D点

    i = 0
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        size = gray.shape[::-1]
        ret, corners = cv2.findChessboardCorners(gray, (11, 8), None)
        #print(fname)
        #print(corners)
        #print('ret', ret)

        if ret: #ret:

            obj_points.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)  # 在原角点的基础上寻找亚像素角点
            #print(corners2)
            if [corners2]:
                img_points.append(corners2)
            else:
                img_points.append(corners)

            cv2.drawChessboardCorners(img, (11, 8), corners, ret)  # 记住，OpenCV的绘制函数一般无返回值
            i+=1
            #cv2.imwrite('conimg'+str(i)+'.jpg', img)
            #cv2.waitKey(1500)

    #print(len(img_points),len(obj_points),size)
    cv2.destroyAllWindows()

    # 标定
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, size, None, None)

    print("ret:", ret)
    print("mtx:\n", mtx) # 内参数矩阵
    print("dist:\n", dist)  # 畸变系数   distortion cofficients = (k_1,k_2,p_1,p_2,k_3)
    print("rvecs:\n", rvecs)  # 旋转向量  # 外参数
    print("tvecs:\n", tvecs ) # 平移向量  # 外参数

    print("-----------------------------------------------------")
    return mtx, dist

def calibrate(img):
    #hand cam 1080*1920 params
    #mtx=np.array( [[652.39226579,   0.,         522.61617923],
    #                [  0.,         653.3911041,  979.11112664],
    #                [  0.,           0.,           1.        ]])
    #dist=np.array( [[ 0.07104035, -0.05197842,  0.00156287,  0.00024595,  0.00771525]])

    #hand cam 1080*1920 params
    mtx=np.array([[444.58569763,   0.,         661.77146694],
                    [  0.,         445.07620211, 369.12141815],
                    [  0.,           0.,           1.        ]])
    dist=np.array([[ 0.04128792, -0.00758873, -0.00043866,  0.00183363, -0.01251645]])
    
    #img = cv2.resize(img,(321,181))

    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))#显示更大范围的图片（正常重映射之后会删掉一部分图像）
    #print(newcameramtx)
    #print("------------------使用undistort函数-------------------")
    dst = cv2.undistort(img,mtx,dist,None,newcameramtx)
    x,y,w,h = roi
    dst1 = dst[y:y+h, x:x+w]
    #cv2.imwrite('test_calibrate_imgs/calibresult2.jpg', dst1)
    #cv2.imwrite('test_calibrate_imgs/calibresult2-src.jpg', dst)
    print("方法一:dst的大小为:", dst1.shape)
    print("方法一:dst的大小为:", dst.shape)
    return dst1

if __name__ == '__main__':
    #cal_img_path = "hand_cam_calibrate_imgs_1280_720/*.jpg"
    #mtx, dist = get_cam_dist(cal_img_path)
    test_img_path = 'hand_cam_calibrate_imgs_1280_720/1756440731.8597658.jpg'
    img = cv2.imread(test_img_path)
    calibrate(img)