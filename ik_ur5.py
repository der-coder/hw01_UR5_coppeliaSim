# -*- coding: utf-8 -*-
"""
Created on Mon May 25 04:30:27 2020

@author: Isaac Ayala
"""

# Load coppeliaSim library
try:
    import sim
except:
    print ('--------------------------------------------------------------')
    print ('"sim.py" could not be imported. This means very probably that')
    print ('either "sim.py" or the remoteApi library could not be found.')
    print ('Make sure both are in the same folder as this file,')
    print ('or appropriately adjust the file "sim.py"')
    print ('--------------------------------------------------------------')
    print ('')

# Load other libraries for delays and specifying coordinates as multiples of pi
import time
import math as m
import numpy as np

def ur5_params():
    """
    Returns
    -------
    home_position : Home position of the UR5 with DH parameters
    screw : Screw matrix for UR5 with DH parameters
    """
    # UR5 link parameters (in meters)
    l1 = 0.425
    l2 = 0.392
    h1 = 0.160
    h2 = 0.09475
    w1 = 0.134
    w2 = 0.0815

    home_position = np.array([
      [-1, 0, 0, l1 + l2],
      [0, 0, 1, w1 + w2],
      [0, 1, 0, h1 - h2],
      [0, 0, 0, 1]
    ])

    screw = np.array([
      [0,0,1,0,0,0],
      [0,1,0,-h1,0,0],
      [0,1,0,-h1, 0,l1],
      [0,1,0, -h1, 0, l1+l2],
      [0,0,-1, -w1, l1+l2, 0],
      [0,1,0, h2-h1, 0, l1+l2]
    ])
    return home_position, screw

def crossProductOperator(vector):
    w1,w2,w3 = vector
    W = [
        [  0, -w3,  w2],
        [ w3,   0, -w1],
        [-w2,  w1,   0]
        ]
    Ax = np.asarray(W)
    return Ax

def exponential_map(action_axis, theta):
    action_axis = np.asarray(action_axis)
    linear = action_axis[3:]
    angular = action_axis[:3]

    exp_rot = exponential_form_rotation(angular, theta)
    exp_tras = exponential_form_traslation(linear, angular, theta)

    expMap = np.block([[exp_rot, exp_tras], [np.zeros( (1,3) ), 1]])
    return(expMap)

def exponential_form_rotation(angular, theta):
    c = crossProductOperator(angular) @ crossProductOperator(angular)
    expRot = np.eye(3) + np.sin(theta) * crossProductOperator(angular) + ( 1-np.cos(theta) ) * c
    return expRot

def exponential_form_traslation(linear, angular, theta):
    l1, l2, l3 = linear
    lin = np.array([[l1, l2, l3]])
    angular_mat = crossProductOperator(angular)
    c = angular_mat  @ angular_mat
    expTras = (theta * np.eye(3) + ( 1 - np.cos(theta) ) * angular_mat + ( theta - np.sin(theta) ) * c) @ (lin.transpose())
    return expTras

def Ad(mat4):
    mat4  = np.asarray(mat4)
    rot = mat4[:3, :3]
    tras = mat4[0:3, 3]
    Ad = np.block([[rot, crossProductOperator(tras) @ rot],[np.zeros((3,3)), rot]])
    return(Ad)

def denavit_transformation(theta, index):
    """
    Computes the homogeneous transformation according to the Denavit-Hatenberg convention

    Parameters
    ----------
    theta : Rotation in z-axis [radians].

    Internal variables
    ------------------
    d : Distance between x-axes in meters.
    alpha : Angle between z_1 and and z_0 axes.
    r : Distance between z-axes.

    Returns
    -------
    G : Homogeneous transformation.

    """

    d = [0.089159, 0, 0, 0.10915, 0.09465, 0.0523]
    alpha = [m.pi/2, 0, 0, m.pi/2, -m.pi/2, 0]
    r = [0, -0.425, -0.3922, 0, 0, 0]

    c_theta = m.cos(theta)
    s_theta = m.sin(theta)
    c_alpha = m.cos(alpha[index])
    s_alpha = m.sin(alpha[index])

    # print('DH Values: ', c_theta, s_theta, c_alpha, s_alpha)

    R_z = np.array([
                    [c_theta, -s_theta, 0, 0],
                    [s_theta, c_theta, 0, 0],
                    [0, 0, 1 ,0],
                    [0, 0, 0, 1]
                    ]) # DH rotation z-axis
    T_z = np.array([
                    [1, 0 ,0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1 , d[index]],
                    [0, 0, 0, 1]
                    ]) # DH translation z-axis
    T_x = np.array([
                    [1, 0, 0, r[index]],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]
                    ]) # DH translation x-axis
    R_x = np.array([
                    [1, 0, 0, 0],
                    [0, c_alpha, -s_alpha, 0],
                    [0, s_alpha, c_alpha, 0],
                    [0, 0, 0, 1]
                    ]) # DH rotation x-axis

    # print(R_z, T_z, T_x, R_x)

    G = R_z @ T_z @ T_x @ R_x

    return G

def compute_jacobian(theta, screw, dof=6 ):
    # Compute DH transformations
    # Compute exponential maps too

    G_local = []
    expMap_local = []
    for i in range(dof):
        G_local.append(denavit_transformation(theta[i], i))
        expMap_local.append(G_local[i] @ exponential_map(screw[i], theta[i]))

    # Get G for the adjoint operator
    G = []
    for i in range(dof):
        if i == 0:
            g = np.eye(4)
        else:
            g = g @ G[i-1]
        G.append(g @ expMap_local[i])

    # Get space Jacobian
    J_s = []
    for i in range(6):
        J_s.append(Ad(G[i]) @ screw[i])

    # print('Space : \n', J_s)

    # Get location of end effector tip (p_tip)
    p_k = np.zeros((3,1))
    # p_k = np.array([[0],[0],[0.5]])

    p_0_extended = G[5] @ np.block([[p_k],[1]])
    p_0 = p_0_extended[:3]

    p_0_cpo = np.array([[0, -p_0[2], p_0[1]],[p_0[2], 0, -p_0[0]],[-p_0[1], p_0[0], 0]])

        # Geometric Jacobian
    """

    The geometric Jacobian is obtained from the spatial Jacobian and the vector p_tip

    p_tip : tip point of the end effector
    p_0^tip : p measured from the inertial reference frame
    p_k^tip : p measured from the frame k, if k is the end effector and the tip is at its origin then p_k = 0

    [p_0^tip; 1] = G_0^k [p_k^tip; 1]

    """
    J_g = np.block([[np.eye(3), -p_0_cpo],[np.zeros((3,3)), np.eye(3)]]) @ J_s

    # print('Geometric : \n', J_g)

    # Get Roll, Pitch, Yaw coordinates
    R = G[5][0:3][0:3]

    r_roll = m.atan2(R[2][1],R[2][2])
    r_pitch = m.atan2(-R[2][0],m.sqrt(R[2][2]*R[2][2] + R[2][1]*R[2][1]))
    r_yaw = m.atan2(R[1][0],R[0][0])

    # Build kinematic operator for Roll, Pitch, Yaw configuration
    # Taken from Olguin's formulaire book

    B = np.array(
        [
            [m.cos(r_pitch) * m.cos(r_yaw), -m.sin(r_yaw), 0],
            [m.cos(r_pitch) * m.cos(r_yaw), m.cos(r_yaw), 0],
            [- m.sin(r_pitch), 0, 1]
            ]
        )

    # print('Kinematic : \n', B)

    # Get Analytic Jacobian
    """

    Obtained from function

    J_a(q) = [[I 0],[0, B(alpha)^-1]] J_g(q)

    B(alpha) =    for roll, pitch, yaw

    """
    J_a = np.block(
        [
            [np.eye(3), np.zeros((3, 3))],
            [np.zeros((3,3)), np.linalg.inv(B)]
            ]
        )

    return J_a

def compute_e(theta_d, theta_0, dof, home_position, screw):
    T_0 = compute_T(screw, theta_0, dof, home_position)
    T_d = compute_T(screw, theta_d, dof, home_position)
    e = T_d - T_0

    # e = theta_d - theta_0
    return e

def root_finding(theta_0, theta_d, tryMax, dof, home_position, screw):

    n_try = 1; # Count number of iterations
    tol = 0.0001; # error tolerance
    theta = theta_0
    e = compute_e(theta_d, theta, dof, home_position, screw) # Gets error from the transformation matrix

    while n_try < tryMax and np.linalg.norm(e) > tol :
        ja = compute_jacobian(theta, screw)
        j_temp = np.zeros((6,6))
        for i in range(6):
            for j in range (6):
                j_temp[i][j] = ja[i][j]

        inverse_jacobian = np.linalg.inv(j_temp)
        theta = theta + inverse_jacobian @ (theta_d - theta)
        e = compute_e(theta_d, theta, dof, home_position, screw)
        n_try += 1
    return theta, e, n_try


def compute_T(screw, theta, dof, M):
    """


    Parameters
    ----------
    screw : screw matrix.
    theta : coordinates.
    dof : robot degrees of freedom.
    M : 4 x 4 matrix describing the position of the end effector at theta_i = 0.

    Returns
    -------
    T : New end effector position.

    """

    expMap_local = []
    T = np.eye(4)
    for i in range(dof):
        expMap_local.append(exponential_map(screw[i], theta[i]))
        T = T @ expMap_local[i]
    T = T @ M
    return T

def main():
    np.set_printoptions(precision=3, suppress=True) # Restrict decimals in console output

    dof = 6
    home_position, screw = ur5_params() # Get UR5 parameters

    theta_0 = np.array([0, 0, 0, 0, 0, 0]) # Initial position
    theta_d = np.array([0,-m.pi/2, 0, 0, m.pi/2, 0]) # Desired position, converted to x_d in the solver

    T_0 = compute_T(screw, theta_0, dof, home_position)
    T_d = compute_T(screw, theta_d, dof, home_position)

    print("Home position: \n", T_0, "\n")
    print("Desired position: \n", T_d, "\n")

    # Find solution to the system
    theta, delta, n = root_finding(theta_0, theta_d, 20, dof, home_position, screw)
    T_theta = compute_T(screw, theta, dof, home_position)

    print('Solution : \n', theta, '\n', 'Transformation : \n', T_theta, '\n')

    R = T_theta[0:3][0:3] # Get RPY for the solution

    r_roll = m.atan2(R[2][1],R[2][2])
    r_pitch = m.atan2(-R[2][0],m.sqrt(R[2][2]*R[2][2] + R[2][1]*R[2][1]))
    r_yaw = m.atan2(R[1][0],R[0][0])

    # Begin connection with coppeliaSim.
    # Robot simulation must be running in coppeliaSim to connect
    sim.simxFinish(-1) # just in case, close all opened connections
    clientID=sim.simxStart('127.0.0.1',19997,True,True,5000,5) # Connect to CoppeliaSim

# clientID stores the ID assingned by coppeliaSim, if the connection failed it will be assigned -1
    if clientID!=-1:
        print ('Connected to remote API server')

# Get ID handles for each joint of the robot
        res,UR5_joint1 =sim.simxGetObjectHandle( clientID, 'UR5_joint1',    sim.simx_opmode_blocking)
        res,UR5_joint2 =sim.simxGetObjectHandle( clientID, 'UR5_joint2', sim.simx_opmode_blocking)
        res,UR5_joint3 =sim.simxGetObjectHandle( clientID, 'UR5_joint3', sim.simx_opmode_blocking)
        res,UR5_joint4 =sim.simxGetObjectHandle( clientID, 'UR5_joint4', sim.simx_opmode_blocking)
        res,UR5_joint5 =sim.simxGetObjectHandle( clientID, 'UR5_joint5', sim.simx_opmode_blocking)
        res,UR5_joint6 =sim.simxGetObjectHandle( clientID, 'UR5_joint6', sim.simx_opmode_blocking)

        res,UR5_endEffector =sim.simxGetObjectHandle( clientID, 'UR5_connection', sim.simx_opmode_blocking)

# Store the handles as a list
    UR5 = [UR5_joint1, UR5_joint2, UR5_joint3, UR5_joint4, UR5_joint5, UR5_joint6] # Just grab the first 3 for now to test

# Get current coordinates of each joint
    UR5_q = []
    for joint in UR5:
        res, value = sim.simxGetJointPosition(clientID, joint, sim.simx_opmode_oneshot)
        UR5_q.append(value) # Store current values

# Set new coordinates for the robot in coppeliaSim
# Add time delays so the animation can be displayed correctly
    steps = 100
    position_desired = theta

    for t in range(steps):
        i = 0
        k = 0
        for joint in UR5:
            sim.simxSetJointTargetPosition(clientID, joint, t*(position_desired[i])/steps, sim.simx_opmode_streaming)
            res, value = sim.simxGetJointPosition(clientID, joint, sim.simx_opmode_oneshot)
            if t == 0 or t == 99:
                k += 1
                res, position = sim.simxGetObjectPosition(clientID, UR5_endEffector, UR5_joint1,  sim.simx_opmode_oneshot)
                res, orientation = sim.simxGetObjectOrientation(clientID, UR5_endEffector, UR5_joint1,  sim.simx_opmode_oneshot)
            i += 1
            time.sleep(2/steps)

    # Convert robot angles to the 4x4 matrix representation for comparison
    c1, s1 = np.cos(orientation[0]), np.sin(orientation[0])
    c2, s2 = np.cos(orientation[1]), np.sin(orientation[1])
    c3, s3 = np.cos(orientation[2]), np.sin(orientation[2])

    R_ur5 = np.block([
        [c2 * c3, -c2 * s3, s2],
        [c1 * s3 + c3 * s1 * s2, c1 * c3 - s1 * s2 * s3, -c2 * s1],
        [s1 * s3 - c1 * c3 * s2, c3 * s1 + c1 * s2 * s3, c1 * c2]
        ])

    p_ur5 = np.array(position).reshape((3,1))

    T_ur5 = np.block([
        [R_ur5, p_ur5],
        [0, 0, 0, 1]
        ])

    print('\n Robot coordinates: \n ', T_ur5 )

    # print('\n Absolute Error : \n', abs(T_theta - T_ur5))

    time.sleep(1)

    # print('\n Returning to home position...')
    # for joint in UR5:
            # sim.simxSetJointTargetPosition(clientID, joint, 0, sim.simx_opmode_streaming)

    # Now send some data to CoppeliaSim in a non-blocking fashion:
    sim.simxAddStatusbarMessage(clientID,'Excuting forward kinematics in Python',sim.simx_opmode_oneshot)

    # Before closing the connection to CoppeliaSim, make sure that the last command sent out had time to arrive. You can guarantee this with (for example):
    sim.simxGetPingTime(clientID)

    # Now close the connection to CoppeliaSim:
    sim.simxFinish(clientID)


if __name__ == '__main__':
        main()
