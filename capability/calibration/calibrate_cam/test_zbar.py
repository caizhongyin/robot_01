import cv2
from pyzbar.pyzbar import decode

def decode_barcode(image):
    # 解码找到的图像中的条形码
    barcodes = decode(image)
    
    # 遍历所有检测到的条形码
    for barcode in barcodes:
        # 提取条形码的边界框坐标
        (x, y, w, h) = barcode.rect
        # 在图像上绘制矩形框
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)
        
        # 条形码数据和解码类型
        barcode_data = barcode.data.decode("utf-8")
        barcode_type = barcode.type
        
        # 在图像上绘制条形码类型和数据
        text = f"{barcode_data} ({barcode_type})"
        cv2.putText(image, text, (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 打印信息到控制台
        print(f"发现条形码: 类型 {barcode_type}, 数据: {barcode_data}")
    
    return image

if __name__ == '__main__':
    # 使用示例
    img_path = "./images/1756441650.8428118.jpg"
    image = cv2.imread(img_path)
    result_image = decode_barcode(image)
    cv2.imwrite(img_path.replace('.jpg','_zbar.jpg'), result_image)
    #cv2.imshow("Barcode Detection", result_image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()