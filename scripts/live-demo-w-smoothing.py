import os
import sys
import argparse
import ast
import cv2
import time
import torch
from vidgear.gears import CamGear
import numpy as np
import random
import scipy
from scipy import signal

sys.path.insert(1, os.getcwd())
from SimpleHRNet import SimpleHRNet
from misc.visualization import draw_points, draw_skeleton, draw_points_and_skeleton, joints_dict, check_video_rotation
from misc.utils import find_person_id_associations
from misc.kalman import KF2d

# Temporal utility functions
joint_buffer_size = 13
# MS COCO, 17 joints
num_of_joints = 17
#joint_buffer = [np.random.rand(num_of_joints,3).astype('f') for i in range(joint_buffer_size)]
joint_buffer = []
#print (joint_buffer)

##########################################################################################################
#
# Testing some smoothing methods
#
##########################################################################################################
def add_joint_points_to_buffer(points):
    joint_buffer.append(points)
    #print (joint_buffer[-joint_buffer_size:])
    return joint_buffer[-joint_buffer_size:]

# average filtering
def simplest_filter(input_buffer):
    #print(input_buffer)
    #print (np.mean(input_buffer, axis=0))
    return np.mean(input_buffer, axis=0)

# Apply a Savitzky-Golay filter to an array.
# output shape (17,3)
def savgol_filter(input_buffer):
    output_buffer = []
    window_length, polyorder = joint_buffer_size, 2
    savgol_out = np.zeros((num_of_joints, 3))
    for i in range(joint_buffer_size): output_buffer.append(savgol_out)

    #print (output_buffer)
    #print (input_buffer[0])
    # print nose 
    #print (input_buffer[0][0][0])
    #print (input_buffer[0][1])
    for i in range(num_of_joints):
        each_joint = np.zeros((joint_buffer_size, 3))
        for j in range(joint_buffer_size):
            #print (j,i)
            #print (input_buffer)
            each_joint[j][0] = input_buffer[j][i][0]
            each_joint[j][1] = input_buffer[j][i][1]
            each_joint[j][2] = input_buffer[j][i][2]
        #print (each_joint[:,0])
        #print ('-------------')
        #print (signal.savgol_filter(each_joint[:,0], window_length, polyorder))
        for k, (x, y, c) in enumerate(zip(signal.savgol_filter(each_joint[:,0], window_length, polyorder),
                signal.savgol_filter(each_joint[:,1], window_length, polyorder),
                signal.savgol_filter(each_joint[:,2], window_length, polyorder))):
            #print (k, x, y, c)
            output_buffer[k][i][0] = x
            output_buffer[k][i][1] = y
            output_buffer[k][i][2] = c

    return output_buffer

def point2xyv(kp):
    kp = kp.flatten()
    kp = np.array(kp)
    x = kp[0::3].astype(int)
    y = kp[1::3].astype(int)
    v = kp[2::3].astype(int) # visibility, 0 = Not visible, 0 != visible
    return x,y,v

list_KFs = []
def kalman_filter_init():
    for i in range(num_of_joints):
        KF = KF2d( dt = 1 ) # time interval: '1 frame'
        init_P = 1*np.eye(4, dtype=np.float) # Error cov matrix
        init_x = np.array([0,0,0,0], dtype=np.float) # [x loc, x vel, y loc, y vel]
        dict_KF = {'KF':KF,'P':init_P,'x':init_x}
        list_KFs.append(dict_KF)

def kalman_filter(points):

    kx,ky,kv = point2xyv(points) # x, y, visiblity
    #print (kx, ky, kv)

    list_estimate = [] # kf filtered keypoints
    cnt_validpoint = 0
    start = cv2.getTickCount()
    for i in range(num_of_joints):
        z = np.array( [kx[i], ky[i]], dtype=np.float)
        KF = list_KFs[i]['KF']
        x  = list_KFs[i]['x']
        P  = list_KFs[i]['P']
        
        x, P, filtered_point = KF.process(x, P, z)
        
        list_KFs[i]['KF'] = KF
        list_KFs[i]['x']  = x
        list_KFs[i]['P']  = P
        
        # visibility
        v = 0 if filtered_point[0] == 0 and filtered_point[1] == 0 else 1
        list_estimate.extend(list(filtered_point) + [v]) # x,y,v

    time = (cv2.getTickCount() - start) / cv2.getTickFrequency() * 1000 # ms
    print ('[INFO] %d kfs aver time: %.2fms'%(num_of_joints, time/num_of_joints))
   
    print (list_estimate)

    points = np.reshape(list_estimate, (num_of_joints,3))
    print (points)

    return points

##########################################################################################################

def main(camera_id, filename, hrnet_m, hrnet_c, hrnet_j, hrnet_weights, hrnet_joints_set, image_resolution,
         single_person, use_tiny_yolo, disable_tracking, max_batch_size, disable_vidgear, save_video, video_format,
         video_framerate, device):
    if device is not None:
        device = torch.device(device)
    else:
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

    # print(device)

    image_resolution = ast.literal_eval(image_resolution)
    has_display = 'DISPLAY' in os.environ.keys() or sys.platform == 'win32'
    video_writer = None

    if filename is not None:
        rotation_code = check_video_rotation(filename)
        video = cv2.VideoCapture(filename)
        assert video.isOpened()
    else:
        rotation_code = None
        if disable_vidgear:
            video = cv2.VideoCapture(camera_id)
            assert video.isOpened()
        else:
            video = CamGear(camera_id).start()

    if use_tiny_yolo:
         yolo_model_def="./models/detectors/yolo/config/yolov3-tiny.cfg"
         yolo_class_path="./models/detectors/yolo/data/coco.names"
         yolo_weights_path="./models/detectors/yolo/weights/yolov3-tiny.weights"
    else:
         yolo_model_def="./models/detectors/yolo/config/yolov3.cfg"
         yolo_class_path="./models/detectors/yolo/data/coco.names"
         yolo_weights_path="./models/detectors/yolo/weights/yolov3.weights"

    model = SimpleHRNet(
        hrnet_c,
        hrnet_j,
        hrnet_weights,
        model_name=hrnet_m,
        resolution=image_resolution,
        multiperson=not single_person,
        return_bounding_boxes=not disable_tracking,
        max_batch_size=max_batch_size,
        yolo_model_def=yolo_model_def,
        yolo_class_path=yolo_class_path,
        yolo_weights_path=yolo_weights_path,
        device=device
    )

    if not disable_tracking:
        prev_boxes = None
        prev_pts = None
        prev_person_ids = None
        next_person_id = 0

    # initialize kalman filter
    kalman_filter_init()
    frame_cnt = 0
    while True:
        t = time.time()

        if filename is not None or disable_vidgear:
            ret, frame = video.read()
            if not ret:
                break
            if rotation_code is not None:
                frame = cv2.rotate(frame, rotation_code)
        else:
            frame = video.read()
            if frame is None:
                break

        pts = model.predict(frame)

        if not disable_tracking:
            boxes, pts = pts

        if not disable_tracking:
            if len(pts) > 0:
                if prev_pts is None and prev_person_ids is None:
                    person_ids = np.arange(next_person_id, len(pts) + next_person_id, dtype=np.int32)
                    next_person_id = len(pts) + 1
                else:
                    boxes, pts, person_ids = find_person_id_associations(
                        boxes=boxes, pts=pts, prev_boxes=prev_boxes, prev_pts=prev_pts, prev_person_ids=prev_person_ids,
                        next_person_id=next_person_id, pose_alpha=0.2, similarity_threshold=0.4, smoothing_alpha=0.1,
                    )
                    next_person_id = max(next_person_id, np.max(person_ids) + 1)
            else:
                person_ids = np.array((), dtype=np.int32)

            prev_boxes = boxes.copy()
            prev_pts = pts.copy()
            prev_person_ids = person_ids

        else:
            person_ids = np.arange(len(pts), dtype=np.int32)

        for i, (pt, pid) in enumerate(zip(pts, person_ids)):
            #np.save('pt.npy', pt)
            #cur_joints_buf = add_joint_points_to_buffer(pt)

            # for savgol_filter
            #if (frame_cnt > joint_buffer_size * 6):
            # for the simplest fitler (moving average)
            #if (frame_cnt > joint_buffer_size * 6):
              #print (frame_cnt)
              #print (frame_cnt, joint_buffer_size)
              #print ('------------')
              #print (len(cur_joints_buf))
              #pt = simplest_filter(cur_joints_buf)
              #pt_out = savgol_filter(cur_joints_buf)
              #pt = pt_out[-1]
              #print (pt)

            # for kalman filter=
            pt = kalman_filter(pt)

            frame = draw_points_and_skeleton(frame, pt, joints_dict()[hrnet_joints_set]['skeleton'], person_index=pid,
                                             points_color_palette='gist_rainbow', skeleton_color_palette='jet',
                                             points_palette_samples=10)

        fps = 1. / (time.time() - t)
        print('\rframerate: %f fps' % fps, end='')
        frame_cnt += 1
        #print('frame count: %d' %frame_cnt)

        if has_display:
            cv2.imshow('frame.png', frame)
            k = cv2.waitKey(1)
            if k == 27:  # Esc button
                if disable_vidgear:
                    video.release()
                else:
                    video.stop()
                break
        else:
            cv2.imwrite('frame.png', frame)

        if save_video:
            if video_writer is None:
                fourcc = cv2.VideoWriter_fourcc(*video_format)  # video format
                video_writer = cv2.VideoWriter('output.avi', fourcc, video_framerate, (frame.shape[1], frame.shape[0]))
            video_writer.write(frame)

    if save_video:
        video_writer.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera_id", "-d", help="open the camera with the specified id", type=int, default=0)
    parser.add_argument("--filename", "-f", help="open the specified video (overrides the --camera_id option)",
                        type=str, default=None)
    parser.add_argument("--hrnet_m", "-m", help="network model - 'HRNet' or 'PoseResNet'", type=str, default='HRNet')
    parser.add_argument("--hrnet_c", "-c", help="hrnet parameters - number of channels (if model is HRNet), "
                                                "resnet size (if model is PoseResNet)", type=int, default=48)
    parser.add_argument("--hrnet_j", "-j", help="hrnet parameters - number of joints", type=int, default=17)
    parser.add_argument("--hrnet_weights", "-w", help="hrnet parameters - path to the pretrained weights",
                        type=str, default="./weights/pose_hrnet_w48_384x288.pth")
    parser.add_argument("--hrnet_joints_set",
                        help="use the specified set of joints ('coco' and 'mpii' are currently supported)",
                        type=str, default="coco")
    parser.add_argument("--image_resolution", "-r", help="image resolution", type=str, default='(384, 288)')
    parser.add_argument("--single_person",
                        help="disable the multiperson detection (YOLOv3 or an equivalen detector is required for"
                             "multiperson detection)",
                        action="store_true")
    parser.add_argument("--use_tiny_yolo",
                        help="Use YOLOv3-tiny in place of YOLOv3 (faster person detection). Ignored if --single_person",
                        action="store_true")
    parser.add_argument("--disable_tracking",
                        help="disable the skeleton tracking and temporal smoothing functionality",
                        action="store_true")
    parser.add_argument("--max_batch_size", help="maximum batch size used for inference", type=int, default=16)
    parser.add_argument("--disable_vidgear",
                        help="disable vidgear (which is used for slightly better realtime performance)",
                        action="store_true")  # see https://pypi.org/project/vidgear/
    parser.add_argument("--save_video", help="save output frames into a video.", action="store_true")
    parser.add_argument("--video_format", help="fourcc video format. Common formats: `MJPG`, `XVID`, `X264`."
                                                     "See http://www.fourcc.org/codecs.php", type=str, default='MJPG')
    parser.add_argument("--video_framerate", help="video framerate", type=float, default=30)
    parser.add_argument("--device", help="device to be used (default: cuda, if available)."
                                         "Set to `cuda` to use all available GPUs (default); "
                                         "set to `cuda:IDS` to use one or more specific GPUs "
                                         "(e.g. `cuda:0` `cuda:1,2`); "
                                         "set to `cpu` to run on cpu.", type=str, default=None)
    args = parser.parse_args()
    main(**args.__dict__)
