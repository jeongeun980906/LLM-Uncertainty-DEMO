import time
import rclpy
from sensor_msgs.msg import JointState
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from rclpy.action import ActionClient
from rclpy.node import Node
from math import pi
import numpy as np
import sys
from builtin_interfaces.msg import Duration
from model.mujoco_parser import MuJoCoParserClass
from model.util import sample_xyzs,rpy2r,r2quat,get_interp_const_vel_traj
from pymodbus.client.sync import ModbusTcpClient
from model.gripper import *
import copy

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

Q1 = capture_pose_q



class RobotClient(Node):

    def __init__(self):
        super().__init__('minimal_client_async')
        self.joint_state_subscriber = self.create_subscription(
            JointState,
            "/joint_states",
            self.joint_state_callback,
            10,
        )
        action_name = "/scaled_joint_trajectory_controller/follow_joint_trajectory"
        action_type = FollowJointTrajectory
        self.action_client = self.waitForAction(action_name, action_type)
        try:
            self.params = self.get_parameters(["/tf_prefix"])
            self.tf_prefix = self.params[0].get_parameter_value().string_value
        except:
            self.tf_prefix = ""
        print(self.tf_prefix)
        self.capture_q = np.array([np.deg2rad(-90), np.deg2rad(-132.46), np.deg2rad(122.85), np.deg2rad(99.65), np.deg2rad(45), np.deg2rad(-90.02)])
        self.init_sim()
        self.graspclient = ModbusTcpClient('192.168.0.13') 
        self.get_prepose()

    def joint_state_callback(self, msg):
        joint_names = msg.name
        self.current_q = np.zeros(6)
        self.current_qdot = np.zeros(6)
        for i, joint_name in enumerate(joint_names):
            indx = ROBOT_JOINTS.index(joint_name)
            self.current_q[indx] = msg.position[i]
            self.current_qdot[indx] = msg.velocity[i]

    def init_sim(self):
        xml_path = './asset/visualize_realworld_wo_shelf_pringles.xml'
        env = MuJoCoParserClass(name='Place task scene: Office table',rel_xml_path=xml_path,VERBOSE=False, MODE='window')
        print(env.MODE)

        # Move tables and robot base
        env.model.body('base_table').pos = np.array([0,0,0])
        env.model.body('avoiding_object_table').pos = np.array([0.38+0.45,0,0])
        env.model.body('ur_base').pos = np.array([0.18,0,0.79])
        env.model.body('right_object_table').pos = np.array([-0.05,-0.80,0])
        env.model.body('left_object_table').pos = np.array([-0.05,0.80,0])

        joint_names = env.rev_joint_names[:6]
        self.idxs_forward = [env.model.joint(joint_name).qposadr[0] for joint_name in env.joint_names[:6]]
        self.idxs_jacobian = [env.model.joint(joint_name).dofadr[0] for joint_name in env.joint_names[:6]]
        list1, list2 = env.ctrl_joint_idxs, self.idxs_forward
        idxs_step = []
        for i in range(len(list2)):
            if list2[i] in list1:
                idxs_step.append(list1.index(list2[i]))
        self.env = env

    def call_action(self, action_name, goal):
        self.get_logger().info(f"Sending goal to action server '{action_name}'")
        future = self.action_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)

        if future.result() is not None:
            return future.result()
        else:
            raise Exception(f"Exception while calling action: {future.exception()}")

    def get_prepose(self):
        self.env.init_viewer(viewer_title='IK solver',viewer_width=1200,viewer_height=800,viewer_hide_menus=True, MODE='window')
        self.env.update_viewer(azimuth=00,distance=2.5,elevation=-30,lookat=[0,0,1.5])
        self.env.update_viewer(VIS_JOINT=False,jointlength=0.5,jointwidth=0.1,jointrgba=[0.2,0.6,0.8,0.6])
        self.env.reset() # reset
        # print(self.env.get_q())
        q_init = self.capture_q.copy()
        self.env.forward(q=q_init,joint_idxs=self.idxs_forward)
        p_target = np.array([0.73, 0.0 , 0.83])
        R_target = rpy2r(np.array([-180,0,90])*np.pi/180.0)

        q_ik_target = self.solve_ik(p_trgt=p_target,R_trgt=R_target,
            IK_P=True,IK_R=True, q_init=q_init,
            idxs_forward=self.env.idxs_forward, idxs_jacobian=self.env.idxs_jacobian,
            inc_prefix = 'ur', exc_prefix=None,
            RESET=False, DO_RENDER=True, th=1, err_th=1e-3, stepsize=0.5, w_weight=0.3, render_every=1.0, repulse=5)
        print(f"Initial Joint values: {q_init}")
        print(f"Solved IK: {q_ik_target}")
        # Close viewer
        print ("Done.")
        self.env.viewer.close()
        self.pre_pose = q_ik_target

    def send_request(self, trajectory):
        goal_response = self.call_action(
            "/scaled_joint_trajectory_controller/follow_joint_trajectory",
            FollowJointTrajectory.Goal(trajectory=trajectory),
        )
        while True:
            rclpy.spin_once(self, timeout_sec=0.1)
            # self.get_logger().info("Result of action: {}".format(goal_response))
            if goal_response.status ==4:
                break
        return goal_response
    
    def waitForAction(self, action_name, action_type, timeout=TIMEOUT_WAIT_ACTION):
        client = ActionClient(self, action_type, action_name)
        if client.wait_for_server(timeout) is False:
            raise Exception(
                f"Could not reach action server '{action_name}' within timeout of {timeout}"
            )

        self.get_logger().info(f"Successfully connected to action '{action_name}'")
        return client

    def reset_capture_pose(self):
        test_trajectory = [
                (Duration(sec=7, nanosec=0), self.capture_q),
                # (Duration(sec=6, nanosec=500000000), [-0.1 for j in ROBOT_JOINTS]),
            ]
        print(test_trajectory)
        trajectory = JointTrajectory(
            joint_names=[self.tf_prefix + joint for joint in ROBOT_JOINTS],
            points=[
                JointTrajectoryPoint(positions=test_pos, time_from_start=test_time)
                for (test_time, test_pos) in test_trajectory
            ],
        )
        self.send_request(trajectory)
        openGrasp(400,1000,self.graspclient)

    def infeasible_callback(self, deg = 30):
        deg = 30

        q_array = np.vstack([self.current_q, self.pre_pose])
        self.send_trajectory(q_array, vel=30)
        time.sleep(2)
        capture_pose_q = self.capture_q.copy()
        rad = deg/180.0*np.pi
        left = capture_pose_q.copy()
        left[0] += rad
        right = capture_pose_q.copy()
        right[0] -= rad
        q_array = np.vstack([self.pre_pose, capture_pose_q, left, right])
        for _ in range(1):
            q_array = np.vstack([q_array, left, right])
        q_array = np.vstack([q_array, capture_pose_q])
        self.send_trajectory(q_array, vel=25)

    def reset_from_back(self):
        q_pos = self.current_q.copy()
        mid_point = q_pos.copy()
        mid_point[0] = np.deg2rad(0)
        q_array = np.vstack([q_pos, mid_point])
        capture_pose_q = self.capture_q.copy()
        q_array = np.vstack([q_array, capture_pose_q])
        self.send_trajectory(q_array, vel=25)


    def pick(self, p_target):
        
        self.env.init_viewer(viewer_title='IK solver',viewer_width=1200,viewer_height=800,viewer_hide_menus=True, MODE='window')
        self.env.update_viewer(azimuth=00,distance=2.5,elevation=-30,lookat=[0,0,1.5])
        self.env.update_viewer(VIS_JOINT=False,jointlength=0.5,jointwidth=0.1,jointrgba=[0.2,0.6,0.8,0.6])
        self.env.reset() # reset

        q_init = self.pre_pose.copy()
        self.env.forward(q=q_init,joint_idxs=self.idxs_forward)
        p_target = p_target #np.array([0.65, 0.27 , 0.9])
        R_target = rpy2r(np.array([-180,0,90])*np.pi/180.0)


        q_ik_target = self.solve_ik(p_trgt=p_target,R_trgt=R_target,
            IK_P=True,IK_R=True, q_init=q_init,
            idxs_forward=[0,1,2,3,4], idxs_jacobian=[0,1,2,3,4],
            inc_prefix = 'ur', exc_prefix=None,
            RESET=False, DO_RENDER=True, th=1, err_th=1e-3, stepsize=1, w_weight=0.3, render_every=1.0, repulse=5)
        print(f"Initial Joint values: {q_init}")
        print(f"Solved IK: {q_ik_target}")
        q_ik_target = np.hstack([q_ik_target,q_init[-1]])
        # Close viewer
        print ("Done.")
        # q_array = np.vstack([q_init,q_ik_target])
        # self.send_trajectory(q_array)
        ## move 2cm forward
        openGrasp(400,1000,self.graspclient)
        p_target2 = p_target + np.array([0.07,0,0]) 
        q_ik_target2 = self.solve_ik(p_trgt=p_target2,R_trgt=R_target,
            IK_P=True,IK_R=True, q_init=q_ik_target,
            idxs_forward=[0,1,2,3,4], idxs_jacobian=[0,1,2,3,4],
            inc_prefix = 'ur', exc_prefix=None,
            RESET=False, DO_RENDER=True, th=1, err_th=1e-3, stepsize=1, w_weight=0.3, render_every=1.0, repulse=5)
        q_ik_target2 = np.hstack([q_ik_target2,q_ik_target[-1]])
        q_array = np.vstack([self.current_q, q_init, q_ik_target])
        self.send_trajectory(q_array, vel=20)
        time.sleep(0.5)
        q_array = np.vstack([q_ik_target, q_ik_target2])
        self.send_trajectory(q_array, vel=10)
        time.sleep(1)
        closeGrasp(200,500,self.graspclient)

        self.env.close_viewer()
    def give(self, id=0):
        self.env.init_viewer(viewer_title='IK solver',viewer_width=1200,viewer_height=800,viewer_hide_menus=True, MODE='window')
        self.env.update_viewer(azimuth=00,distance=2.5,elevation=-30,lookat=[0,0,1.5])
        self.env.update_viewer(VIS_JOINT=False,jointlength=0.5,jointwidth=0.1,jointrgba=[0.2,0.6,0.8,0.6])
        self.env.reset() # reset
        # move up 10cm
        current_q = self.current_q
        self.env.forward(q=current_q,joint_idxs=self.idxs_forward)
        current_p = self.env.get_p_body(body_name='ur_tcp_link')
        print(current_p,current_q)
        p_target = np.array([current_p[0], current_p[1] , 1.0])
        R_target = rpy2r(np.array([-180,0,90])*np.pi/180.0)
        q_init = current_q.copy()
        q_ik_target = self.solve_ik(p_trgt=p_target,R_trgt=R_target,
            IK_P=True,IK_R=True, q_init=q_init,
            idxs_forward=[0,1,2,3,4], idxs_jacobian=[0,1,2,3,4],
            inc_prefix = 'ur', exc_prefix=None,
            RESET=False, DO_RENDER=True, th=1, err_th=1e-3, stepsize=1, w_weight=0.3, render_every=1.0, repulse=10)
        print(f"Initial Joint values: {q_init}")
        print(f"Solved IK: {q_ik_target}")
        q_ik_target = np.hstack([q_ik_target,q_init[-1]])
        q_array = np.vstack([q_init,q_ik_target])
        waypoint_1 = np.array([0.65, 0.0, 1.6])
        print(q_ik_target)
        q_ik_target2 = self.solve_ik(p_trgt=waypoint_1,R_trgt=R_target,
            IK_P=True,IK_R=True, q_init=q_ik_target,
            idxs_forward=[0,1,2,3,4], idxs_jacobian=[0,1,2,3,4],
            inc_prefix = 'ur', exc_prefix=None,
            RESET=False, DO_RENDER=True, th=1, err_th=1e-3, stepsize=1, w_weight=0.3, render_every=1.0, repulse=10)
        q_ik_target2 = np.hstack([q_ik_target2,q_ik_target[-1]])
        q_array = np.vstack([q_array,q_ik_target2])
        # Close viewer
        print ("Done.")
        if id == 0:
            # most left
            middle = None #[None]
            p_target = np.array([-0.1, 0.5 , 1.0])
            R_target = rpy2r(np.array([0,180,0])*np.pi/180.0)
        if id == 1:
            # middle left
            middle = 120/180.0*np.pi
            # p_target_middle = np.array([-0.1, 0.3 , 1.6])
            p_target = np.array([-0.5, 0.2 , 1.0])
            R_target = rpy2r(np.array([0,180,90])*np.pi/180.0)
        if id == 2:
            # middle right
            middle = -120/180.0*np.pi
            # p_target_middle = np.array([-0.1, -0.3 , 1.6])
            # R_target_middle = rpy2r(np.array([0,180,180])*np.pi/180.0)
            p_target = np.array([-0.5, -0.2 , 1.0])
            R_target = rpy2r(np.array([0,180,90])*np.pi/180.0)
        if id == 3:
            # most right
            middle = None
            # p_target_middle = [None]
            p_target = np.array([-0.1, -0.5 , 1.0])
            R_target = rpy2r(np.array([0,180,180])*np.pi/180.0)
        
        # if p_target_middle[0] != None:
        #     print("solve the mid point")
        #     q_ik_target_middle = self.solve_ik(p_trgt=p_target_middle,R_trgt=R_target_middle,
        #         IK_P=True,IK_R=True, q_init=q_ik_target2,
        #         idxs_forward=[0,1,2,3,4], idxs_jacobian=[0,1,2,3,4],
        #         inc_prefix = 'ur', exc_prefix=None,
        #         RESET=False, DO_RENDER=True, th=1, err_th=1e-3, stepsize=1, w_weight=0.3, render_every=1.0, repulse=20)
        #     q_ik_target_middle = np.hstack([q_ik_target_middle,q_ik_target[-1]])
        #     q_array = np.vstack([q_array,q_ik_target_middle])
        #     q_init = q_ik_target_middle.copy()
        # else: q_init = q_ik_target2.copy()
        if middle != None:
            q_middle = q_ik_target2.copy()
            q_middle[0] += middle
            q_array = np.vstack([q_array,q_middle])
            q_init = q_middle.copy()
        else: q_init = q_ik_target2.copy()
        # time.sleep(2)
        q_ik_target3 = self.solve_ik(p_trgt=p_target,R_trgt=R_target,
            IK_P=True,IK_R=True, q_init=q_init,
            idxs_forward=[0,1,2,3,4], idxs_jacobian=[0,1,2,3,4],
            inc_prefix = 'ur', exc_prefix=None,
            RESET=False, DO_RENDER=True, th=1, err_th=1e-3, stepsize=1, w_weight=0.3, render_every=1.0, repulse=10)
        q_ik_target3 = np.hstack([q_ik_target3,q_ik_target[-1]])
        # print(q_ik_target2.shape)
        # time.sleep(5)
        self.env.close_viewer()
        q_array = np.vstack([q_array,q_ik_target3])
        self.send_trajectory(q_array, vel=25)
        time.sleep(1)
        openGrasp(400,1000,self.graspclient)


    def send_trajectory(self, q_array, vel=30):
        trajectory = []
        total_time = 0
        for i in range(q_array.shape[0]-1):
            new_array = q_array[i:i+2,:]
            # print(new_array)
            times,_ = get_interp_const_vel_traj(new_array, vel=np.radians(vel), HZ=self.env.HZ)
            # print(times)
            total_time += times[-1]
            sec = int(total_time)
            nanosec = int((total_time - sec) * 1e9)
            dur = Duration(sec=sec, nanosec=nanosec)
            trajectory.append((dur, q_array[i+1,:]))
        # print(trajectory)
        trajectory_req = JointTrajectory(
            joint_names=[self.tf_prefix + joint for joint in ROBOT_JOINTS],
            points=[
                JointTrajectoryPoint(positions=test_pos, time_from_start=test_time)
                for (test_time, test_pos) in trajectory
            ],
        )
        self.send_request(trajectory_req)

    def solve_ik(self,p_trgt,R_trgt,IK_P,IK_R,q_init,idxs_forward, idxs_jacobian,
                RESET=False,DO_RENDER=False,render_every=1,th=1*np.pi/180.0,err_th=1e-6,w_weight=1.0, stepsize=1.0, eps=0.1,
                repulse = 30, inc_prefix = None, exc_prefix = None):
        """
            Solve IK
        """
        if RESET:
            self.env.reset()
        q_backup = self.env.get_q(joint_idxs=idxs_forward)
        q = q_init.copy()
        self.env.forward(q=q,joint_idxs=self.env.idxs_forward)
        q = q[idxs_forward]
        tick = 0
        while True:
            tick = tick + 1
            J,err = self.env.get_ik_ingredients(
                body_name='ur_tcp_link',p_trgt=p_trgt,R_trgt=R_trgt,IK_P=IK_P,IK_R=IK_R, w_weight=w_weight)
            dq = self.env.damped_ls(J,err,stepsize=stepsize,eps=eps,th=th)
            q = q + dq[idxs_jacobian]
            self.env.forward(q=q,joint_idxs=idxs_forward)

            p_contacts,f_contacts,geom1s,geom2s,body1s,body2s = self.env.get_contact_info(must_include_prefix=inc_prefix, must_exclude_prefix=exc_prefix)
            
            body1s_ = [obj_ for obj_ in body1s if obj_ not in ["ur_rg2_gripper_finger1_finger_tip_link","ur_rg2_gripper_finger2_finger_tip_link"]]
            body2s_ = [obj_ for obj_ in body2s if obj_ not in ["ur_rg2_gripper_finger1_finger_tip_link","ur_rg2_gripper_finger2_finger_tip_link"]]
            
            if len(body1s_) > 0:
                q = q - dq[idxs_jacobian] * repulse
            
            # Terminate condition
            err_norm = np.linalg.norm(err)
            if err_norm < err_th:
                break
            # Render
            if DO_RENDER:
                if ((tick-1)%render_every) == 0:
                    p_tcp,R_tcp = self.env.get_pR_body(body_name='ur_tcp_link')
                    self.env.plot_T(p=p_tcp,R=R_tcp,PLOT_AXIS=True,axis_len=0.1,axis_width=0.005)
                    self.env.plot_T(p=p_trgt,R=R_trgt,PLOT_AXIS=True,axis_len=0.2,axis_width=0.005)
                    self.env.render(render_every=render_every)
        # Back to back-uped position
        q_ik = self.env.get_q(joint_idxs=idxs_forward)
        self.env.forward(q=q_backup,joint_idxs=idxs_forward)
        q_ik = rand_in_range(q_ik)
        return q_ik
    
    #### 
def rand_in_range(q):
    cos_q = np.cos(q)
    sin_q = np.sin(q)
    refined_q = np.arctan2(sin_q, cos_q)
    return refined_q