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

def get_line_trajectory(initial_point=[350, -300, 100], final_point=[350, 300, 300], num_points=50):
    # Initial and final points
    initial_point = np.array(initial_point)
    final_point = np.array(final_point)
    # Generate straight line trajectory
    xyz = np.linspace(initial_point, final_point, num_points)
    vector0 = np.zeros((num_points, 2))    
    vector1 = np.ones((num_points, 1))
    return np.hstack((xyz, vector0, -vector1))

def get_circular_trajectory(center=[350, 0, 100], radius=100.0, num_points=50):
# Center and radius of the circle
    center = np.array(center)
    # Angle parameter for the circless
    theta = np.linspace(0, 2 * np.pi, num_points)
    # Generate circular trajectory
    x_circle = center[0] + radius * np.cos(theta)
    y_circle = center[1] + radius * np.sin(theta)
    # z_circle = np.linspace(center[2], center[2]+200, num_points)
    z_circle = center[2] + np.zeros_like(theta)  # Z coordinate remains constant
    vector0 = np.zeros((num_points,1))
    vector1 = np.ones((num_points, 1))
    return np.column_stack((x_circle, y_circle, z_circle, vector0, vector0, -vector1))

def get_helix_trajectory(center=[350, 0, 100], radius=100.0, num_points=80):
# Center and radius of the circle
    center = np.array(center)
    # Angle parameter for the circless
    theta = np.linspace(0, 5 * np.pi, num_points)
    # Generate circular trajectory
    x_circle = center[0] + radius * np.cos(theta)
    y_circle = center[1] + radius * np.sin(theta)
    z_circle = np.linspace(center[2], center[2]+200, num_points)
    # z_circle = center[2] + np.zeros_like(theta)  # Z coordinate remains constant
    vector0 = np.zeros((num_points,1))
    vector1 = np.ones((num_points, 1))
    return np.column_stack((x_circle, y_circle, z_circle, vector0, vector0, -vector1))

lengths = [231.5, 221.12, 0, 223, 170]
min_angles = np.radians([-60, 0, -180, -45, 0])
max_angles = np.radians([60, 90, -90, 45, 180])
arm = RoboticsArm(lengths=lengths, q_low=min_angles, q_high=max_angles)

get_custom_objects().update({"get_total_loss": RoboticsArm.get_total_loss})
get_custom_objects().update({"get_physics_loss": RoboticsArm.get_physics_loss})
get_custom_objects().update({"get_BC_loss": RoboticsArm.get_BC_loss})
get_custom_objects().update({"get_position_loss": RoboticsArm.get_position_loss})
get_custom_objects().update({"get_orientation_loss": RoboticsArm.get_orientation_loss})
model = load_model("C:\\MyFolder\\Code\\Python\\Thesis\\final\\model_12_29_14_42_[0.13155946135520935, 0.18043625354766846, 4.208344398648478e-05]\\model.h5")

choice = input('choice')
if 'line' in choice:
    target_points = get_line_trajectory()
elif 'circle'in choice:
    target_points = get_circular_trajectory()
else:
    target_points = get_helix_trajectory()
print(target_points.shape)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# A = arm.generate_random_data(50)
# target_points, batch_angles = next(A)
predictions = model.predict(target_points)
print('angles:', np.degrees(predictions[0]))
end_effector, _ = arm.batch_forward_kinematics(predictions)
ax.scatter(target_points[:, 0], target_points[:, 1], target_points[:, 2], s=1, color='red', label='Waypoints')
# ax.quiver()
ax.scatter(end_effector[:, 0], end_effector[:, 1], end_effector[:, 2], s=1, color='blue', label='Waypoints')
# ax.quiver(end_effector[:, 0], end_effector[:, 1], end_effector[:, 2], end_effector[:, 3]*50, end_effector[:, 4]*50, end_effector[:, 5]*50, color='green', label='n_x')
for i, prediction in enumerate(predictions):
    arm.plot_robotic_arm(prediction, ax)
ax.set_xlim([-600, 600]) #400->600
ax.set_ylim([-400, 400])
ax.set_zlim([0, 700])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Robotic Arm')
ax.set_box_aspect([1, 1, 1])
ax.legend()
plt.savefig(f'{choice}.png', dpi=300)
plt.show()