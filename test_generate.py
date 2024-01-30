import numpy as np
import tensorflow as tf
from math import pi
from Arm3D.config import RoboticsArm

# Create and use RoboticsArm instance
arm_5DOF = RoboticsArm(
    lengths=(231.5, 221.12, 0, 223, 170),
    q_low=np.radians([-90, 0, -180, -90, 0]),
    q_high=np.radians([90, 180, 0, 90, 180])
    )
# Generate and plot data
end_effector, theta = arm_5DOF.generate_random_data()
