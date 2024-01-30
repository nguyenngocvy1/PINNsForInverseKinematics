from math import *
import numpy as np
from keras.models import load_model
from keras.utils import get_custom_objects
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

from Arm3D.config import RoboticsArm

def get_line_trajectory(initial_point=[350, -300, 100], final_point=[350, 300, 100], num_points=1000):
    # Initial and final points
    initial_point = np.array(initial_point)
    final_point = np.array(final_point)
    # Generate straight line trajectory
    xyz = np.linspace(initial_point, final_point, num_points)
    vector0 = np.zeros((num_points, 2))    
    vector1 = np.ones((num_points, 1))
    return np.hstack((xyz, vector0, -vector1))

def get_circular_trajectory(center=[350, 0, 100], radius=100.0, num_points=1000):
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

def get_helix_trajectory(center=[350, 0, 100], radius=100.0, num_points=1000):
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

def write_distance():
    with open(f"model_12_29_14_42_[0.13155946135520935, 0.18043625354766846, 4.208344398648478e-05]\\distance_{choice}.txt", "w") as file:
        file.write("Loss Distances:\n")
        loss_distances = np.linalg.norm(target_points[:, :3] - end_effectors[:, :3], axis=1)
        num = 2
        print(loss_distances[num])
        print(sqrt((target_points[num, 0] - end_effectors[num, 0])**2 +(target_points[num, 1] - end_effectors[num, 1])**2 +(target_points[num, 2] - end_effectors[num, 2])**2))
        file.write(f"{loss_distances}mm")
        file.write("\n\nAverage Loss Distances:\n")
        file.write(f"{sum(loss_distances)/len(loss_distances)}mm")
        file.write("\n\nMax Loss Distances:\n")
        file.write(f"{max(loss_distances)}mm")
        file.write("\n\nMin Loss Distances:\n")
        file.write(f"{min(loss_distances)}mm")

def write_orientation():
    with open(f"model_12_29_14_42_[0.13155946135520935, 0.18043625354766846, 4.208344398648478e-05]\\orientation_{choice}.txt", "w") as file:
        file.write("Loss Orientation:\n")
        loss_orientations = np.degrees(np.arccos(np.sum(target_points[:, 3:] * end_effectors[:, 3:], axis=1)))
        # cosine_angles = np.clip(cosine_angles, -1.0, 1.0)
        file.write(f"{loss_orientations}")
        file.write("\n\nAverage Loss Orientations:\n")
        file.write(f"{sum(loss_orientations)/len(loss_orientations)}")
        file.write("\n\nMax Loss Distances:\n")
        file.write(f"{max(loss_orientations)}")
        file.write("\n\nMin Loss Distances:\n")
        file.write(f"{min(loss_orientations)}")
        
        

lengths = [231.5, 221.12, 0, 223, 170]
print(17*100/sum(lengths))
print(14*100/180)
min_angles = np.radians([-60, 0, -180, -45, 0])
max_angles = np.radians([60, 90, -90, 45, 90])
arm = RoboticsArm(lengths=lengths, q_low=min_angles, q_high=max_angles)

get_custom_objects().update({"get_total_loss": RoboticsArm.get_total_loss})
get_custom_objects().update({"get_physics_loss": RoboticsArm.get_physics_loss})
get_custom_objects().update({"get_BC_loss": RoboticsArm.get_BC_loss})
get_custom_objects().update({"get_position_loss": RoboticsArm.get_position_loss})
get_custom_objects().update({"get_orientation_loss": RoboticsArm.get_orientation_loss})

model = load_model("C:\\MyFolder\\Code\\Python\\Thesis\\final\\model_12_29_14_42_[0.13155946135520935, 0.18043625354766846, 4.208344398648478e-05]\\model.h5")

choice = input('choice')
if 'Line' in choice:
    target_points = get_line_trajectory()
elif 'Circle' in choice:
    target_points = get_circular_trajectory()
elif 'Helix' in choice:
    target_points = get_helix_trajectory()
else:
    generate = arm.generate_random_data(1000)
    target_points, _ = next(generate)
   
predictions = model.predict(target_points)
# Assuming you have joint_positions as a NumPy array with shape (num_frames, num_joints, 3)
# where num_frames is the number of frames, num_joints is the number of joints, and 3 is the (x, y, z) coordinates.

end_effectors, joint_positions = arm.batch_forward_kinematics(predictions)
write_distance()
write_orientation()

joint_positions = joint_positions.astype(int)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim([-600, 600])  # Adjust these limits based on your data
ax.set_ylim([-400, 400])
ax.set_zlim([0, 700])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title(f'Robotic Arm {choice} Drawing')

for target_point in target_points:
    x, y, z, n_x, n_y, n_z = target_point
    ax.scatter(x, y, z, s=1, color='red')

(line1,) = ax.plot([], [], [], color='blue', label="Arm l1")
(line2,) = ax.plot([], [], [], color='purple', label="Arm l2")
(line3,) = ax.plot([], [], [], color='gray', label="Arm l3")
(line4,) = ax.plot([], [], [], color='green', label="Arm l4")
(line5,) = ax.plot([], [], [], color='black', label="Arm l5")
ax.legend()
# Function to update the plot in each frame
def update(i):
    if i >= len(joint_positions):
        animation.event_source.add_callback(save_work_result)
        animation.event_source.stop()
        return
    # Plot the links of the robotic arm
    line1.set_xdata([0, joint_positions[i, 0, 0]])
    line1.set_ydata([0, joint_positions[i, 0, 1]])
    line1.set_3d_properties([0, joint_positions[i, 0, 2]])

    line2.set_xdata([joint_positions[i, 0, 0], joint_positions[i, 1, 0]])
    line2.set_ydata([joint_positions[i, 0, 1], joint_positions[i, 2, 1]])
    line2.set_3d_properties([joint_positions[i, 0, 2], joint_positions[i, 1, 2]])

    line3.set_xdata([joint_positions[i, 1, 0], joint_positions[i, 2, 0]])
    line3.set_ydata([joint_positions[i, 1, 1], joint_positions[i, 2, 1]])
    line3.set_3d_properties([joint_positions[i, 1, 2], joint_positions[i, 2, 2]])
    

    line4.set_xdata([joint_positions[i, 2, 0], joint_positions[i, 3, 0]])
    line4.set_ydata([joint_positions[i, 2, 1], joint_positions[i, 3, 1]])
    line4.set_3d_properties([joint_positions[i, 2, 2], joint_positions[i, 3, 2]])

    line5.set_xdata([joint_positions[i, 3, 0], joint_positions[i, 4, 0]])
    line5.set_ydata([joint_positions[i, 3, 1], joint_positions[i, 4, 1]])
    line5.set_3d_properties([joint_positions[i, 3, 2], joint_positions[i, 4, 2]])
    if i < joint_positions.shape[0]:
        ax.scatter(end_effectors[i, 0], end_effectors[i, 1], end_effectors[i, 2], s=1, color='blue', label='Waypoints')
    return (line1, line2, line3, line4, line5,)

def save_work_result():
        fig.savefig(f"{choice}_anime.png")
        plt.close(fig)

# Create the animation
animation = FuncAnimation(fig, func=update, frames=joint_positions.shape[0], interval=1)
animation.save(f'anime_{choice}_5DOF.gif', writer='imagemagick', fps=60)
plt.show()