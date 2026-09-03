# handintention
# Hand-Trajectory-Based Human Intention Recognition for UR10e Human–Robot Collaboration

## Overview

This project develops a hand-trajectory-based human intention recognition system for human–robot collaborative pick-and-place tasks.

The main objective is to allow a UR10e collaborative robot to understand which object a human intends to interact with and then select a different object to avoid task conflict.

A top-view camera is used to capture the human hand movement in the workspace. The hand intention recognition algorithm analyses several motion and spatial features, including the hand-to-object distance, finger direction, hand motion direction, trajectory information, and approach behaviour.

After the human target is identified, the predicted target information is transmitted to the MuJoCo simulation through UDP communication. The UR10e robot then uses this information to select a non-conflicting object and perform the corresponding pick-and-place operation.

The project contains separate modules for UR10e motion testing, hand intention recognition, and laboratory environment simulation.


## Project Structure

### `ur10e_project/`

This folder is mainly used to test the UR10e robot and verify its motion behaviour in the MuJoCo simulation environment.

It contains the basic simulation environment for testing robot states, joint movements, end-effector motion, and pick-and-place operations.

This module should be deployed and executed under:

- Ubuntu 22.04
- MuJoCo
- Python

The purpose of this module is to verify that the UR10e robot can correctly perform the required motion before integrating it with the human intention recognition system.


### `hand_intention/`

This folder contains the main hand intention recognition algorithm.

A top-view camera is used to observe the human hand and objects in the workspace. The system tracks the hand trajectory and estimates the object that the human intends to reach.

The intention recognition algorithm considers multiple features instead of simply selecting the nearest object. These features include:

- Hand-to-object distance
- Finger pointing direction
- Hand motion direction
- Hand trajectory
- Approach behaviour
- Temporal stability

After a target is confirmed, the predicted human target information is transmitted through UDP to the UR10e program running in the MuJoCo environment.

The robot then avoids selecting the same target as the human and performs a pick-and-place operation on another available object.

The communication process can be summarised as:

Human Hand Movement  
↓  
Hand Tracking  
↓  
Trajectory and Feature Analysis  
↓  
Human Target Prediction  
↓  
UDP Communication  
↓  
MuJoCo UR10e Simulation  
↓  
Non-conflicting Object Selection  
↓  
Robot Pick-and-Place


### `ur10e/`

This folder contains the simulated laboratory environment.

The environment is designed to reproduce the experimental setup used for the human–robot collaboration task, including the UR10e robot, workspace, objects, and pick-and-place areas.

This simulation is used to demonstrate the complete collaborative task and evaluate the interaction between the human intention recognition system and the robot.


## System Workflow

The overall system consists of two main components:

1. **Human Intention Recognition**

   The camera captures the human hand movement from a top-view perspective. The algorithm analyses the hand trajectory and related spatial and directional features to estimate the intended target object.

2. **UR10e Robot Control in MuJoCo**

   Once the human target is confirmed, the target information is sent to the MuJoCo environment through UDP. The UR10e robot receives this information and selects another available object to avoid conflict with the human.

The overall objective is not only to detect the human hand, but also to predict human intention before the object is actually grasped. This allows the robot to respond earlier and improves the coordination of the collaborative task.


## Requirements

The UR10e MuJoCo simulation is designed to run under:

- Ubuntu 22.04
- Python
- MuJoCo

The hand intention recognition module requires a camera for real-time hand tracking.

Additional Python packages may be required depending on the specific script, such as OpenCV, MediaPipe, NumPy, and MuJoCo Python packages.


## UDP Communication

UDP communication is used to connect the hand intention recognition module with the MuJoCo UR10e simulation.

The basic communication process is:

`Hand Intention Recognition → UDP → MuJoCo → UR10e Robot Control`

When the human target is confirmed, the corresponding target information is sent to the robot simulation. The robot then updates its object selection strategy and performs a non-conflicting pick-and-place task.


## Project Objective

The main objective of this project is to investigate whether hand trajectory information can be used to improve human intention recognition in collaborative robotic tasks.

Compared with a simple nearest-distance method, the proposed system combines multiple motion and directional features together with temporal stability to provide a more reliable prediction of human intention.

The final system demonstrates how early human intention recognition can be integrated with a UR10e collaborative robot to reduce task conflicts during shared pick-and-place operations.

