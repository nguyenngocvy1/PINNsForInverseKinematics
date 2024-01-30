import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib.animation import FuncAnimation
from keras.models import load_model
from keras.utils import get_custom_objects
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

def dh_transform_matrix(theta, d, a, alpha):
    ct = np.cos(theta)
    st = np.sin(theta)
    ca = np.cos(alpha)
    sa = np.sin(alpha)
    transform = np.array([
        [ct, -st * ca, st * sa, a * ct],
        [st, ct * ca, -ct * sa, a * st],
        [0, sa, ca, d],
        [0, 0, 0, 1],
    ])
    return transform

# class RoboticsArm:
#     def __init__(self, lengths, q_low, q_high):
#         self.lengths = lengths
#         self.num_links = len(lengths)
#         self.max_range = sum(self.lengths)
#         self.q_low = q_low
#         self.q_high = q_high
#         self.d = (  lengths[0],         0, lengths[2], lengths[3],         0)  # link offset
#         self.a = (          0, lengths[1],         0,         0, lengths[4])  # link length
#         self.alpha = (np.pi/2,         0,  -np.pi/2,   np.pi/2,         0)  # link twist

#     def forward_kinematics(self, theta):
#         T = np.eye(4)
#         link_matrices = [np.copy(T)]
        
#         for i in range(self.num_links):
#             T_i = dh_transform_matrix(theta[i], self.d[i], self.a[i], self.alpha[i])
#             T = np.dot(T, T_i)
#             link_matrices.append(np.copy(T))
#         x, y, z = T[:3, 3]
#         roll = np.arctan2(T[2, 1], T[2, 2])
#         pitch = np.arctan2(-T[2, 0], np.sqrt(T[2, 1] ** 2 + T[2, 2] ** 2))
#         yaw = np.arctan2(T[1, 0], T[0, 0])
#         matrix = np.hstack((x, y, z, roll, pitch, yaw))
#         return matrix, link_matrices
    
#     def build_continuous_data(self, step):
#         joint_ranges = zip(self.q_low, self.q_high)
#         q_values = [np.arange(low, high + step, step) for low, high in joint_ranges]
#         q_meshgrid = np.meshgrid(*q_values, indexing='ij')
#         q_flat = [q_i.ravel() for q_i in q_meshgrid]
#         return np.column_stack(q_flat)

    
#     def plot_robotic_arm(self, theta, ax):
#         _, link_matrices = self.forward_kinematics(theta)
#         # Plotting
#         ax.set_xlim([-self.max_range, self.max_range])  # Adjust based on your arm dimensions
#         ax.set_ylim([-self.max_range, self.max_range])  # Adjust based on your arm dimensions
#         ax.set_zlim([0, self.max_range])
#         ax.set_box_aspect([1, 1, 1])
#         # Plot each link separately
#         for i in range(0, self.num_links):
#             x_vals = [link_matrices[i][0, 3], link_matrices[i+1][0, 3]]
#             y_vals = [link_matrices[i][1, 3], link_matrices[i+1][1, 3]]
#             z_vals = [link_matrices[i][2, 3], link_matrices[i+1][2, 3]]
#             ax.plot(x_vals, y_vals, z_vals, linewidth=5, label=f'Link {i+1}')
#         # Plot end-effector
#         ax.scatter([link_matrices[-1][0, 3]], [link_matrices[-1][1, 3]], [link_matrices[-1][2, 3]],
#                    color='k', marker='o', s=100, label='End-Effector')
#         # Set axis labels
#         ax.set_xlabel('X')
#         ax.set_ylabel('Y')
#         ax.set_zlabel('Z')
#         ax.set_title('Robotic Arm')
#         ax.legend()
#         def get_end_effector_loss(self, y_true, y_pred):
#         end_effector_true = self.tf_batch_forward_kinematics(y_true)
#         end_effector_pred = self.tf_batch_forward_kinematics(y_pred)
#         loss = tf.reduce_mean(tf.square(end_effector_true - end_effector_pred))
#         return loss

#     def get_BC_loss(self, y_true, y_pred):
#         loss_per_joint = tf.reduce_mean(
#             tf.square(tf.maximum(self.q_low - y_pred, 0.0))
#             + tf.square(tf.maximum(y_pred - self.q_high, 0.0)),
#             axis=0
#         )
#         loss = tf.reduce_mean(loss_per_joint)
#         return loss

#     def get_total_loss(self, y_true, y_pred):
#         end_effector_loss = tf.cast(self.get_end_effector_loss(y_true, y_pred), tf.float32)
#         BC_loss = tf.cast(self.get_BC_loss(y_true, y_pred), tf.float32)
#         return end_effector_loss + 1e9 * BC_loss

# Test the function with example joint angles
lengths = [231.5, 221.12, 0, 223, 170]
min_angles = np.radians([-90, 0, -180, -180, 0])
max_angles = np.radians([90, 180, 0, 180, 180])
arm = RoboticsArm(lengths=lengths, q_low=min_angles, q_high=max_angles)
batch_size = 10
pos = 0
A = arm.generate_random_data(batch_size)
batch_end_effector, batch_angles = next(A)
end_effector, angles = batch_end_effector[pos], batch_angles[pos]
# # Set up the animation
# data = arm.build_continuous_data(step=5)
# num_frames = data.shape[0]
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# animation = FuncAnimation(fig, update, frames=num_frames, interval=100, repeat=False)

# plt.show()
get_custom_objects().update({"get_total_loss": RoboticsArm.get_total_loss})
get_custom_objects().update({"get_physics_loss": RoboticsArm.get_physics_loss})
get_custom_objects().update({"get_BC_loss": RoboticsArm.get_BC_loss})
model = load_model("C:\\MyFolder\\Code\\Python\\Thesis\\final\\model_12_22_1_22_[0.030565626919269562, 0.01228332705795765, 1.621697811060585e-05]\\model.h5")
with open("C:\\MyFolder\\Code\\Python\\Thesis\\final\\s1.txt", 'r') as f:
    contents = f.readline()
    print('contents: ', contents)
    float_list = [float(value) for value in contents.split()]
    print(np.linalg.norm(np.array(float_list[3:])))
    float_list[3:] = [x*(-1) for x in float_list[3:]]
    float_list[0] += 500
    float_list[2] += 100
# float_list = list(end_effector)
# float_list, _ = arm.forward_kinematics(np.radians([0, 90, -180, 0, 0]))
position = np.array(float_list[:3])
legal_vector = np.array(float_list[3:])
print('end effector: ', type(float_list))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(position[0], position[1], position[2], color='red', label='Waypoint')

# Plot the legal vector as an arrow starting from the waypoint
ax.quiver(position[0], position[1], position[2], legal_vector[0]*100, legal_vector[1]*100, legal_vector[2]*100, color='blue', label='Legal Vector')

prediction = model.predict([float_list])[0]
print('angles:', prediction)
arm.plot_robotic_arm(prediction, ax)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Robotic Arm')
ax.set_box_aspect([1, 1, 1])
ax.legend()
plt.show()