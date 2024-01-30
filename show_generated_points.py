import numpy as np
import tensorflow as tf
from math import pi
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Arm3D.config import RoboticsArm
            
arm_5DOF = RoboticsArm(
    lengths=(231.5, 221.12, 0, 223, 170),
    q_low=np.radians([-60, 0, -180, -45, 0]),
    q_high=np.radians([60, 90, -90, 45, 90])
    )

generator = arm_5DOF.generate_random_data(1_000)
# old_generator = arm_5DOF.old_generate_random_data(1024)
end_effector, theta = next(generator)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(end_effector[:, 0], end_effector[:, 1], end_effector[:, 2], marker='o', s=1, label='End Effector Points')
ax.quiver(end_effector[:, 0], end_effector[:, 1], end_effector[:, 2], end_effector[:, 3]*20, end_effector[:, 4]*20, end_effector[:, 5]*20, color='green', label='n_x')
# for i, v in enumerate(theta):
#     arm_5DOF.plot_robotic_arm(theta[i], ax)
ax.set_box_aspect([1, 1, 1])
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
ax.set_title('3D End Effector Points')
ax.legend()
plt.grid(True)
plt.savefig('3D_End_Effector_Points.png', dpi=300)
plt.show()

# zero_y_points = end_effector[abs(end_effector[:, 1]) < 2]

# Plotting 2D projection on the XY-plane (y = 0)
# plt.scatter(zero_y_points[:, 0], zero_y_points[:, 2], marker='o', s=1, label='End Effector Points (y = 0)')
# plt.quiver(zero_y_points[:, 0], zero_y_points[:, 2], zero_y_points[:, 3]*20, zero_y_points[:, 5]*20, color='green', label='n_x')
# plt.xlabel('X-axis')
# plt.ylabel('Z-axis')
# plt.title('2D Projection on XZ-plane (y = 0)')
# plt.legend()
# plt.grid(True)
# plt.axis('equal')
# plt.savefig('2D_Z_End_Effector_Points.png', dpi=300)
# plt.show()