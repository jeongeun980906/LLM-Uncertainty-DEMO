import numpy as np
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from vision.owl import owl_vit
from sensor_msgs.msg import Image, CameraInfo
import message_filters
from vision.util import get_det_points, Rotation_X, Rotation_Y, Translation, HT_matrix
import math
import cv2
from vision.from_image import gpt4_v_helper
# from vision.detector import compute_xyz
def compute_xyz(depth_img, fx, fy, px, py, height, width):
    indices = np.indices((height, width), dtype=np.float32).transpose(1,2,0)
    z_e = depth_img
    x_e = (indices[..., 1] - px) * z_e / fx
    y_e = (indices[..., 0] - py) * z_e / fy
    xyz_img = np.stack([x_e, y_e, z_e], axis=-1) # Shape: [H x W x 3]
    return xyz_img

class ImageListener(Node):
    def __init__(self, queries, person=False, yolo=False):
        super().__init__('image_listener')
        if yolo: self.model = gpt4_v_helper()
        else:
            self.model = owl_vit()
        self.yolo = yolo
        self.object_queries = queries
        self.model.query_text(queries)
        print("finish loading model")
        self.cv_bridge = CvBridge()
        info_sub = message_filters.Subscriber(self, CameraInfo, '/camera/color/camera_info')
        rbg_sub = message_filters.Subscriber(self, Image, '/camera/color/image_raw')
        depth_sub = message_filters.Subscriber(self, Image, '/camera/aligned_depth_to_color/image_raw')
        ts = message_filters.ApproximateTimeSynchronizer([info_sub, rbg_sub, depth_sub], 10, 1, allow_headerless=True)
        ts.registerCallback(self.callback)
        self.depth_cv = None
        self.rgb_cv = None
        self.det_vis_pub = self.create_publisher(Image, '/det_vis', 10)
        if person:
            self.cap = cv2.VideoCapture(6)
            self.human_cv = None
            if not self.cap.isOpened():
                print("Cannot open camera")
                exit()
            # self.human_cv = None
            # self.human_sub = self.create_subscription(Image, '/zed/image_raw', self.human_callback, 10)
            self.human_det_pub = self.create_publisher(Image, '/human_det_vis', 10)
    def human_callback(self, msg):
        self.human_cv = self.cv_bridge.imgmsg_to_cv2(msg)
        print("got human image")

    def capture_human(self):
        ret, frame = self.cap.read()
        if not ret:
            self.human_cv = None
        else: 
            # frame =cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.human_cv = frame

    def callback(self, info_msg, rgb_msg, depth_msg):
        # camera info
        print(info_msg)
        intrinsics = np.array(info_msg.p).reshape(3, 4)
        self.fx = intrinsics[0, 0]
        self.fy = intrinsics[1, 1]
        self.px = intrinsics[0, 2]
        self.py = intrinsics[1, 2]
        self.intrinsics = intrinsics
        print("got image")
        if depth_msg.encoding == '32FC1':
            depth_cv = self.cv_bridge.imgmsg_to_cv2(depth_msg)
        elif depth_msg.encoding == '16UC1':
            depth_cv = self.cv_bridge.imgmsg_to_cv2(depth_msg).copy().astype(np.float32)
            depth_cv /= 1000.0
        else:
            self.get_logger().error('Depth image has unsupported encoding: %s' % depth_msg.encoding)
            return
        rbg_cv = self.cv_bridge.imgmsg_to_cv2(rgb_msg)
        self.depth_cv = depth_cv
        self.rgb_cv = rbg_cv

    def detect_human(self, human_queries, save_indx = None):
        if not self.yolo:
            self.model.query_text(human_queries)
            if self.human_cv is None:
                print("no human image")
                return None
            print("start detecting")
            if save_indx is not None:
                cv2.imwrite("./data/det/human_ori_{}.jpg".format(save_indx), self.human_cv)
            # inp = cv2.cvtColor(self.human_cv, cv2.COLOR_BGR2RGB)
            im_color,_, boxes, found_object_names = self.model.detect(self.human_cv, thres=0.1)
            if save_indx is not None:
                im_color = cv2.cvtColor(im_color, cv2.COLOR_BGR2RGB)
                cv2.imwrite("./data/det/human_det_{}.jpg".format(save_indx), im_color)
            self.human_det_pub.publish(self.cv_bridge.cv2_to_imgmsg(im_color))
            return boxes, found_object_names
        else:
            file_path = "./data/det/human_ori_{}.jpg".format(save_indx)
            cv2.imwrite(file_path, self.human_cv)
            found_object_name,boxes,_ = self.model.detection(file_path, human_queries, True)
            return boxes, found_object_names
        
    def detect_object(self, save_indx = None):
        self.model.query_text(self.object_queries)
        if self.depth_cv is None or self.rgb_cv is None:
            print("no image")
            return None, None
        print("start detecting")
        inp = self.rgb_cv #cv2.cvtColor(self.rgb_cv, cv2.COLOR_BGR2RGB)
        if save_indx is not None:
            cv2.imwrite("./data/det/ori_{}.jpg".format(save_indx), inp)
        xyz_img = compute_xyz(self.depth_cv, self.fx, self.fy, self.px, self.py, self.depth_cv.shape[0], self.depth_cv.shape[1])
        if not self.yolo:
            im_color,index_mask, boxes, found_object_names = self.model.detect(inp, thres=0.25)
            self.det_vis_pub.publish(self.cv_bridge.cv2_to_imgmsg(im_color))
        else:  found_object_names, _, mask = self.model.detection("./data/det/ori_{}.jpg".format(save_indx), self.object_queries)
        objects_xyz = self.get_position(index_mask, xyz_img)
        if save_indx is not None:
            # im_color = cv2.cvtColor(im_color, cv2.COLOR_BGR2RGB)
            cv2.imwrite("./data/det/det_{}.jpg".format(save_indx), im_color)
        return objects_xyz, found_object_names
    
    def get_position(self, label_mask, xyz_img):
        # Calibration
        rotation_x   = Rotation_X(-math.pi)
        rotation_y   = Rotation_Y(-math.pi/4)
        rotation_mat = np.dot(rotation_x, rotation_y)
        position_mat = Translation(0.511,0.01,1.33)
        transform_mat= HT_matrix(rotation_mat, position_mat)
        objects_xyz = get_det_points(depth_pixel=xyz_img, label_pixel=label_mask, transform_mat=transform_mat)
        return objects_xyz