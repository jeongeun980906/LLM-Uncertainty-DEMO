import rclpy, signal, sys
import pyzed.sl as sl
import numpy as np
import time
from rclpy.node import Node
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

class ZedCam(Node):
    def __init__(self):
        super().__init__('zed_cam')
        self.bridge = CvBridge()
        self.pub = self.create_publisher(Image, '/zed/image_raw', 10)
        self.zed = sl.Camera()
        self.init_params = sl.InitParameters()
        self.init_params.camera_resolution = sl.RESOLUTION.HD720
        # Open the camera
        err = self.zed.open(self.init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            print(repr(err))
            exit(1)
        # Get ZED camera information
        camera_info = self.zed.get_camera_information()
        self.image = sl.Mat()
        self.display_resolution = sl.Resolution(min(camera_info.camera_configuration.resolution.width, 1280), min(camera_info.camera_configuration.resolution.height, 720))
        self.image_scale = [self.display_resolution.width / camera_info.camera_configuration.resolution.width
                    , self.display_resolution.height / camera_info.camera_configuration.resolution.height]
        self.timer_period = 1/30  # seconds
        self.timer = self.create_timer(self.timer_period, self.timer_callback)
    
    def timer_callback(self):
        if self.zed.grab() == sl.ERROR_CODE.SUCCESS:
            # print('grab')
            # Retrieve left image
            self.zed.retrieve_image(self.image, sl.VIEW.LEFT, sl.MEM.CPU, self.display_resolution)
            image_left_ocv = self.image.get_data()
            image_left_ocv = image_left_ocv[:,:,:-1].astype(np.uint8)
            self.pub.publish(self.bridge.cv2_to_imgmsg(image_left_ocv, "bgr8"))


if __name__ == '__main__':
    rclpy.init()
    zed_cam = ZedCam()
    rclpy.spin(zed_cam)
    zed_cam.destroy_node()
    rclpy.shutdown()