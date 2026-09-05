import argparse
import glob
import sys
import time
import cv2
import os
from datetime import datetime
from test_zbar import decode_barcode
from calibrate_cam import calibrate
from img_aug import *

def identify_camera(w=1280, h=720, fps=30, save_video=True, concat=False, cam_num=3, ignore_devices=["0"]):
    if sys.platform.startswith("win"):
        print("Windows系统")
        video_devices = range(cam_num+1)
        sys_name = "windows"
        print("尝试摄像头索引:", video_devices)
    elif sys.platform.startswith("linux"):
        print("Linux系统")
        video_devices = sorted(glob.glob("/dev/video*"))
        sys_name = "linux"
        print("检测到的摄像头设备:", video_devices)
    else:
        exit("未知操作系统")

    if save_video:
        dt_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        os.makedirs(dt_str)
        writers = {}
    caps = {}
    for device in video_devices:
        cap = cv2.VideoCapture(device)
        if cap.isOpened():
            if sys_name == "linux":
                device = f"{int(device.split('video')[-1])}"
            if device in ignore_devices:
                continue
            caps[device] = cap
            fourcc = cv2.VideoWriter_fourcc(*"MJPG") #mp4v XVID MJPG
            cap.set(cv2.CAP_PROP_FOURCC, fourcc)
            cap.set(cv2.CAP_PROP_FPS, fps)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            
            if save_video:
                video_writer = cv2.VideoWriter(dt_str+'/'+dt_str+'_'+device+'.mp4', fourcc, fps, (w, h)) #(w, h) (h, w)
                writers[device] = video_writer
            print(f"成功打开设备: {device}")
        else:
            print(f"无法打开设备: {device}")
            cap.release()

    while True and caps:
        id = 0
        concat_img = None
        for cam_id, cap in caps.items():
            ret, frame = cap.read()
            if ret:
                if frame is not None:
                    h1,w1 = frame.shape[:2]
                    print('cam_id:', cam_id, h1, w1)
                else:
                    continue
                
                #if h1 < w1:
                #    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                #    h1,w1 = frame.shape[:2]
                #frame = calibrate(frame)
                frame = RandomGray(frame)
                #ret, frame = cv2.threshold(frame, 127, 255, cv2.THRESH_BINARY)
                #result_image = decode_barcode(frame)
                result_image = frame
              
                if save_video:
                    writers[cam_id].write(result_image)
                
                if not concat:
                    #cv2.namedWindow(f"Camera {cam_id}", cv2.WINDOW_NORMAL) dt_str
                    #cv2.imwrite(dt_str+"/"+str(time.time())+".jpg", frame)
                    cv2.imshow(f"Camera {cam_id}", result_image)
                else:
                    if id == 0:
                        concat_img = frame
                    else:
                        concat_img = cv2.hconcat([concat_img, frame])
                id += 1
            else:
                break
        if concat_img is not None:
            cv2.imshow(f"concat_img", concat_img)
            
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    for cam_id, cap in caps.items():
        cap.release()
        if save_video:
            writers[cam_id].release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Identify cameras")
    parser.add_argument("--w", type=int, default=1280, help="Width of the camera feed")
    parser.add_argument("--h", type=int, default=720, help="Height of the camera feed")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second")
    parser.add_argument("--save_video", type=bool, default=True, help="Frames per second")
    parser.add_argument("--concat", type=bool, default=False, help="concat images for imshow")
    parser.add_argument("--cam_num", type=int, default=3, help="camera nums")
    parser.add_argument("--ignore_devices", type=list, default=[], help="laptop should skip device 0")

    args = parser.parse_args()
    identify_camera(args.w, args.h, args.fps, args.save_video, args.concat, args.cam_num, args.ignore_devices)
