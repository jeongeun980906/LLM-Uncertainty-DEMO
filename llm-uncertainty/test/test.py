from model.robot_client import *
import rclpy
import numpy as np
import time
from model.robot_client import RobotClient
import json
def main():
    rclpy.init()
    client = RobotClient()
    while True:
        rclpy.spin_once(client, timeout_sec=0.1)
        try: print(client.current_q); break
        except: continue
    client.reset_capture_pose()
    # time.sleep(1)
    # client.pick(np.array([0.90, -0.2 , 0.85]))
    # client.give(id=1)
    # client.reset_from_back()
    # with open("./data/det/demo_{}.json".format(1), "r") as f:
    #     data = json.load(f)
    # objects_xyz = data['objects_xyz']
    # print(objects_xyz[0])
    # client.pick(objects_xyz[2])
    # client.give(3)
    # client.reset_capture_pose()
    client.destroy_node()

if __name__ == '__main__':
    main()