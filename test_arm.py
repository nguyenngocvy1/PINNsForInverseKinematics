import numpy as np
from keras.models import load_model
from keras.utils import get_custom_objects
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

from Arm3D.config import RoboticsArm

def vector_to_euler_angles(vector):
    # Extract components of the vector
    x, y, z = vector
    # Calculate yaw (heading) angle
    yaw = np.arctan2(y, x)
    # Calculate pitch (elevation) angle
    pitch = np.arctan2(-z, np.sqrt(x**2 + y**2))
    # Calculate roll (bank) angle
    roll = np.arctan2(np.sin(yaw) * z - np.cos(yaw) * y, np.cos(yaw) * x - np.sin(yaw) * y)
    return roll, pitch, yaw


lengths = [231.5, 221.12, 0, 223, 170]
min_angles = np.radians([-60, 0, -180, -45, 0])
max_angles = np.radians([60, 90, -90, 45, 90])
arm = RoboticsArm(lengths=lengths, q_low=min_angles, q_high=max_angles)
theta = np.random.uniform(low=min_angles, high=max_angles, size=(100, 5))
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

angles = 40
my_list = [angles, 90-angles, -90-angles, -angles, 90-angles]
for i, v in enumerate(theta):
    arm.plot_robotic_arm(theta[i], ax)
ax.set_xlim([-600, 600])
ax.set_ylim([-600, 600])
ax.set_zlim([0, 800])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Robotic Arm')
ax.set_box_aspect([1, 1, 1])
ax.legend()
plt.savefig('arm.png', dpi=300)
plt.show()