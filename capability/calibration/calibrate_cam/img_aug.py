import random
import cv2
import numpy as np

def RandomGamma(img):
    gamma = 1.5 #random.uniform(0.1,5)
    img = gamma_trans(img, gamma)
    return img
        
def gamma_trans(img, gamma):
    #具体做法先归一化到1，然后gamma作为指数值求出新的像素值再还原
    gamma_table = [np.power(x/255.0,gamma)*255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    #实现映射用的是Opencv的查表函数
    img = cv2.LUT(img, gamma_table)
    return img

def RandomContrast(image):
    alpha = 0.7 #random.uniform(0.7, 1.3)
    np.multiply(image, alpha, out=image, casting="unsafe")
    #image = torch.from_numpy(image).float()
    #image *= alpha
    return image
    
def RandomGray(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_shape = img_gray.shape
    img_gray = img_gray.reshape((img_shape[0],img_shape[1],1))
    img = np.concatenate([img_gray,img_gray,img_gray],axis=2)
    #if random.random() < 0.3:
    #    img = 255 - img
    
    return img

if __name__ == "__main__":
    img_path = './images/zbar_test13.jpg'
    img = cv2.imread(img_path)
    img_gray = RandomGray(img)
    cv2.imwrite(img_path.replace('.jpg', '_gray.jpg'), img_gray)
    img_contrast = RandomContrast(img)
    cv2.imwrite(img_path.replace('.jpg', '_contrast.jpg'), img_contrast)
    img_gamma = RandomGamma(img)
    cv2.imwrite(img_path.replace('.jpg', '_gamma.jpg'), img_gamma)
    ret, binary_image = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
    cv2.imwrite(img_path.replace('.jpg', '_binary.jpg'), binary_image)