import time
# import roslib; roslib.load_manifest('ur_driver')
import rclpy
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from rclpy.action import ActionClient
from rclpy.node import Node
from math import pi
import numpy as np
import sys
from builtin_interfaces.msg import Duration

TIMEOUT_WAIT_ACTION = 10
capture_pose_q = [np.deg2rad(-90), np.deg2rad(-132.46), np.deg2rad(122.85), np.deg2rad(99.65), np.deg2rad(45), np.deg2rad(-90.02)]

ROBOT_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

Q1 = capture_pose_q # [2.2,0,-1.57,0,0,0]
Q2 = [i+0.1 for i in capture_pose_q] # [2.2,0,-1.57,0,0,0]
Q2 = [-0.1,-1.0498095222970842,2.5684575230477313 ,1.518647526493884,1.57,
  -0.0015928023538781865] #[np.deg2rad(-90), np.deg2rad(-135.46), np.deg2rad(122.85), np.deg2rad(99.65), np.deg2rad(45), np.deg2rad(-90.02)]
# [1.5,0,-1.57,0,0,0]
Q3 = [-0.1,-1.0498095222970842,2.5684575230477313 ,1.518647526493884,1.57,
  -0.0015928023538781865 ]#[np.deg2rad(-90), np.deg2rad(-138.46), np.deg2rad(122.85), np.deg2rad(99.65), np.deg2rad(45), np.deg2rad(-90.02)]
#[1.5,-0.2,-1.57,0,0,0]



class MinimalClientAsync(Node):

    def __init__(self):
        super().__init__('minimal_client_async')
        action_name = "/scaled_joint_trajectory_controller/follow_joint_trajectory"
        action_type = FollowJointTrajectory
        self.action_client = self.waitForAction(action_name, action_type)
        try:
            self.params = self.get_parameters(["/tf_prefix"])
            self.tf_prefix = self.params[0].get_parameter_value().string_value
        except:
            self.tf_prefix = ""
        print(self.tf_prefix)

    def call_action(self, action_name, goal):
        self.get_logger().info(f"Sending goal to action server '{action_name}'")
        future = self.action_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)

        if future.result() is not None:
            return future.result()
        else:
            raise Exception(f"Exception while calling action: {future.exception()}")


    def send_request(self, trajectory):
        goal_response = self.call_action(
            "/scaled_joint_trajectory_controller/follow_joint_trajectory",
            FollowJointTrajectory.Goal(trajectory=trajectory),
        )
        self.get_logger().info("Result of action: {}".format(goal_response))
        return goal_response
    
    def waitForAction(self, action_name, action_type, timeout=TIMEOUT_WAIT_ACTION):
        client = ActionClient(self, action_type, action_name)
        if client.wait_for_server(timeout) is False:
            raise Exception(
                f"Could not reach action server '{action_name}' within timeout of {timeout}"
            )

        self.get_logger().info(f"Successfully connected to action '{action_name}'")
        return client


def main(args=None):
    rclpy.init(args=args)
    minimal_client = MinimalClientAsync()
    test_trajectory = [
        (Duration(sec=6, nanosec=0), Q1),
        # (Duration(sec=6, nanosec=500000000), [-0.1 for j in ROBOT_JOINTS]),
    ]
    print(test_trajectory)
    trajectory = JointTrajectory(
        joint_names=[minimal_client.tf_prefix + joint for joint in ROBOT_JOINTS],
        points=[
            JointTrajectoryPoint(positions=test_pos, time_from_start=test_time)
            for (test_time, test_pos) in test_trajectory
        ],
    )
    print(trajectory)
    while True:
        rclpy.spin_once(minimal_client, timeout_sec=0.1)
        inp = input("Continue? y/n: ")[0]
        if (inp == 'y'):
            minimal_client.send_request(trajectory)
            break
    minimal_client.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()