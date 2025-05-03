#!/usr/bin/env python3

import tkinter as tk
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from threading import Thread
from std_srvs.srv import Trigger  # Standard service type for triggering actions
from pymoveit2 import MoveIt2
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Pose, Point, Quaternion
from rclpy.node import Node
from geometry_msgs.msg import Pose
from action_msgs.msg import GoalStatus
from my_robot_interfaces.srv import MoveToPose  # Import the custom service type
from my_robot_interfaces.action import MoveToPoseAc  # Import the custom service type
from tf2_ros import Buffer, TransformListener
from sensor_msgs.msg import JointState
import easy_handeye2_msgs as ehm
import easy_handeye2 as hec
from easy_handeye2_msgs.srv import TakeSample, SaveCalibration,ComputeCalibration
import queue
from rclpy.action import ActionClient
import time
from rclpy.context import Context



class Mover:
    def __init__(self,parent_node):
    # Create self.node for this example

        self.moveit_node = Node(
            node_name="ex_pose_goal",
            # namespace='/mover',  # Adding a specific namespace
            # cli_args=["--ros-args", "-r", "__node:=ex_pose_goal_unique"],  # Explicitly set node name via remapping
            # automatically_declare_parameters_from_overrides=False
        )
        self.should_spin = True
        self.executor = rclpy.executors.MultiThreadedExecutor(2)
        #self.executor.add_node(self.moveit_node)
        #self.executor_thread = Thread(target=self.executor.spin, daemon=True, args=())
        self.executor_thread = Thread(target=self.spin_loop, daemon=True, args=())
        time.sleep(1)
        self.should_spin = True
        self.executor_thread.start()
        self.moveit_node.create_rate(1.0).sleep()
        
                
        self.parent_node = parent_node
        # Declare parameters for position and orientation
        self.moveit_node.declare_parameter("position", [0.5, 0.0, 0.25])
        self.moveit_node.declare_parameter("quat_xyzw", [1.0, 0.0, 0.0, 0.0])
        self.moveit_node.declare_parameter("synchronous", True)
        # If non-positive, don't cancel. Only used if synchronous is False
        self.moveit_node.declare_parameter("cancel_after_secs", 0.0)
        # Planner ID
        self.moveit_node.declare_parameter("planner_id", "RRTConnectkConfigDefault")
        # Declare parameters for cartesian planning
        self.moveit_node.declare_parameter("cartesian", False)
        self.moveit_node.declare_parameter("cartesian_max_step", 0.0025)
        self.moveit_node.declare_parameter("cartesian_fraction_threshold", 0.0)
        self.moveit_node.declare_parameter("cartesian_jump_threshold", 0.0)
        self.moveit_node.declare_parameter("cartesian_avoid_collisions", False)

        # Create callback group that allows execution of callbacks in parallel without restrictions
        callback_group = ReentrantCallbackGroup()

        self.arm_joint_names = [
            "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"
        ]
        self.callback_group = ReentrantCallbackGroup()
        self.parent_node.get_logger().info("---------------------------- MOVEIT NODE NAME1: " + self.moveit_node.get_name())
        # Create MoveIt 2 interface
        self.moveit2 = MoveIt2(
            node=self.moveit_node,
            joint_names=self.arm_joint_names,
            base_link_name="base_link",
            end_effector_name="link_6",
            group_name="ar_manipulator",
            callback_group=self.callback_group
        )
        print("-----------------------------MOVEIT NODE NAME2:", self.moveit_node.get_name())
        self.arm_joint_names = [
            "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"
        ]

        self.moveit2.planner_id = "RRTConnectkConfigDefault"
        self.moveit2.max_velocity = 1.0
        self.moveit2.max_acceleration = 1.0
        self.moveit2.planning_time = 5.0  # Timeout in seconds

        # Spin the self.node in background thread(s) and wait a bit for initialization


        # Scale down velocity and acceleration of joints (percentage of maximum)
        self.moveit2.max_velocity = 0.5
        self.moveit2.max_acceleration = 0.5

        # Get parameters
        # position = self.node.get_parameter("position").get_parameter_value().double_array_value
        # quat_xyzw = self.node.get_parameter("quat_xyzw").get_parameter_value().double_array_value
        # synchronous = self.node.get_parameter("synchronous").get_parameter_value().bool_value
        # cancel_after_secs = (
        #     self.node.get_parameter("cancel_after_secs").get_parameter_value().double_value
        # )
        # cartesian = self.node.get_parameter("cartesian").get_parameter_value().bool_value
        # cartesian_max_step = (
        #     self.node.get_parameter("cartesian_max_step").get_parameter_value().double_value
        # )
        # cartesian_fraction_threshold = (
        #     self.node.get_parameter("cartesian_fraction_threshold")
        #     .get_parameter_value()
        #     .double_value
        # )
        # cartesian_jump_threshold = (
        #     self.node.get_parameter("cartesian_jump_threshold")
        #     .get_parameter_value()
        #     .double_value
        # )
        # cartesian_avoid_collisions = (
        #     self.node.get_parameter("cartesian_avoid_collisions")
        #     .get_parameter_value()
        #     .bool_value
        # )

        # Set parameters for cartesian planning
        # self.moveit2.cartesian_avoid_collisions = cartesian_avoid_collisions

        self.moveit2.cartesian_avoid_collisions = False
        self.moveit2.cartesian_jump_threshold = 0.0

        # self.moveit2.cartesian_jump_threshold = cartesian_jump_threshold
       
    def spin_loop(self):
        while self.should_spin:
            self.executor.spin_once(timeout_sec=0.1)

    def MoveArm(self, position, quat_xyzw):
        time.sleep(0.5) #for some reason the arm is not available immediately after the call
        # Get parameters
        # cartesian = self.self.node.get_parameter("cartesian").get_parameter_value().bool_value
        # cartesian_max_step = self.self.node.get_parameter("cartesian_max_step").get_parameter_value().double_value
        # cartesian_fraction_threshold = self.self.node.get_parameter("cartesian_fraction_threshold").get_parameter_value().double_value
        # cartesian_jump_threshold = self.self.node.get_parameter("cartesian_jump_threshold").get_parameter_value().double_value
        # cartesian_avoid_collisions = self.self.node.get_parameter("cartesian_avoid_collisions").get_parameter_value().bool_value

        # Move to pose goal
        # self.moveit2.move_to_pose(
        #     position=position,
        #     quat_xyzw=quat_xyzw,
        #     synchronous=synchronous,
        #     cancel_after_secs=cancel_after_secs,
        #     cartesian=cartesian,
        #     cartesian_max_step=cartesian_max_step,
        #     cartesian_fraction_threshold=cartesian_fraction_threshold,
        #     cartesian_jump_threshold=cartesian_jump_threshold,
        #     cartesian_avoid_collisions=cartesian_avoid_collisions,
        # )
        pose_goal = PoseStamped() 
        pose_goal.header.frame_id = "base_link"
        pose_goal.pose = Pose(position = position, orientation = quat_xyzw)
        print ("starting to move")
        ret_val = False
        try:
            self.moveit2.move_to_pose(pose=pose_goal,
                                        #synchronous=True,
                                        #cancel_after_secs=0.0,
                                        #cartesian=False,
                                        #cartesian_max_step=0.0025,
                                        #cartesian_fraction_threshold=0.0,
                                        #cartesian_jump_threshold=0.0,
                                        #cartesian_avoid_collisions=False,
                                        #planner_id=self.moveit2.planner_id,                                                       
                                    )
            ret_val = self.moveit2.wait_until_executed()
        finally:
            print ("move completed")
        return ret_val