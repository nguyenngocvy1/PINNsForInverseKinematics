from math import pi

import numpy as np
import tensorflow as tf

from Arm3D.config import RoboticsArm

# # Define the batch size
# batch_size = 32

# # Generate random T01 and T12 matrices for a batch of size 32 (you should replace these with actual values)
# T01_batch = np.random.rand(batch_size, 4, 4)
# T12_batch = np.random.rand(batch_size, 4, 4)
# T23_batch = np.random.rand(batch_size, 4, 4)
# T34_batch = np.random.rand(batch_size, 4, 4)
# T45_batch = np.random.rand(batch_size, 4, 4)
# T56_batch = np.random.rand(batch_size, 4, 4)

# # Use np.einsum() for batch matrix multiplication
# T02_batch = np.einsum('...ij,...jk->...ik', T01_batch, T12_batch)
# T03_batch = np.einsum('...ij,...jk->...ik', T02_batch, T23_batch)
# T04_batch = np.einsum('...ij,...jk->...ik', T03_batch, T34_batch)
# T05_batch = np.einsum('...ij,...jk->...ik', T04_batch, T45_batch)
# T06_batch = np.einsum('...ij,...jk->...ik', T05_batch, T56_batch)

# # T02_batch now contains the resulting T02 matrices for each batch element
# # T02_batch[i] is the transformation from coordinate frame 2 to coordinate frame 0 for the i-th batch element

# # Print the first T02 matrix as an example
# print("Example T02 matrix:")
# print(T02_batch[0])

arm_5DOF = RoboticsArm(
        lengths=(231.5, 221.12, 0, 223, 170),
        q_low=np.radians([-180, -45, -225, -180, -45]),
        q_high=np.radians([180, 225, 0, 180, 225])
        )

batch_size=10
pos = 9
A = arm_5DOF.generate_random_data(batch_size)
batch_end_effector_real, batch_angles_real = next(A)
end_effector_real, angles_real = batch_end_effector_real[pos], batch_angles_real[pos]
# print(angles_real)
print('real', end_effector_real)
batch_end_effector_pred, batch_angles_pred = next(A)

# B = arm_5DOF.get_physics_loss(batch_angles_real, batch_angles_pred)
# tf.print(B)
# C = arm_5DOF.get_BC_loss(batch_angles_real, batch_angles_pred)
# tf.print(C)
# D = arm_5DOF.get_total_loss(batch_angles_real, batch_angles_pred)
# tf.print(D)

G, _, _ = arm_5DOF.forward_kinematics(angles_real)
E = arm_5DOF.batch_forward_kinematics(batch_angles_real)
F, _ = arm_5DOF.tf_batch_forward_kinematics(batch_angles_real)
print('1', G)
print('batch', E[pos])
tf.print('tf', F[pos])