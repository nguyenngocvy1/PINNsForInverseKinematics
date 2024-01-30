import numpy as np
import tensorflow as tf

def dh_transform_matrix(theta, d, a, alpha):#theta:rotateZ, d:translateZ, a:translateX, alpha:rotateX
    ct = np.cos(theta)
    st = np.sin(theta)
    ca = np.cos(alpha)
    sa = np.sin(alpha)
    transform = np.array(
        [
            [ct, -st * ca, st * sa, ct * a],
            [st, ct * ca, -ct * sa, st * a],
            [0, sa, ca, d],
            [0, 0, 0, 1],
        ]
    )
    return transform

def batch_dh_transform_matrix(theta, d, a, alpha):
    d = np.full_like(theta, d)
    a = np.full_like(theta, a)
    alpha = np.full_like(theta, alpha)
    
    ct = np.cos(theta)
    st = np.sin(theta)
    ca = np.cos(alpha)
    sa = np.sin(alpha)
    
    transform = np.array([
        [ct, -st * ca, st * sa, ct * a],
        [st, ct * ca, -ct * sa, st * a],
        [np.zeros_like(ct), sa, ca, d],
        [np.zeros_like(ct), np.zeros_like(ct), np.zeros_like(ct), np.ones_like(ct)]
    ]) # transform.shape = (4, 4, batch_size)
    transform = np.transpose(transform, (2, 0, 1)) #transform.shape = (batch_size, 4, 4)
    return transform

@tf.function
def tf_batch_dh_transform_matrix(theta, d, a, alpha):
    theta = tf.cast(theta, dtype=tf.float32)
    d = tf.cast(d, dtype=tf.float32)
    a = tf.cast(a, dtype=tf.float32)
    alpha = tf.cast(alpha, dtype=tf.float32)

    shape = tf.shape(theta)
    d = tf.fill(shape, d)
    a = tf.fill(shape, a)
    alpha = tf.fill(shape, alpha)

    ct = tf.cos(theta)
    st = tf.sin(theta)
    ca = tf.cos(alpha)
    sa = tf.sin(alpha)

    transform = tf.stack([
        [ct, -st * ca, st * sa, ct * a],
        [st, ct * ca, -ct * sa, st * a],
        [tf.zeros_like(sa), sa, ca, d],
        [tf.zeros_like(a), tf.zeros_like(a), tf.zeros_like(a), tf.ones_like(a)]
    ], axis=0)
    transform = tf.transpose(transform, perm=[2, 0, 1])
    return transform

class RoboticsArm:
    def __init__(self, lengths, q_low, q_high):# chieu dai, gioi han goc duoi,tren
        self.length = lengths
        self.num_links = len(lengths)
        self.q_low = tf.constant(q_low, dtype=tf.float32)
        self.q_high = tf.constant(q_high, dtype=tf.float32)
        self.radius = 50
        self.max_range = sum(self.length)
        self.d = (  lengths[0],         0, lengths[2], lengths[3],         0) #link offset
        self.a = (          0, lengths[1],         0,         0, lengths[4]) #link length
        self.alpha = (np.pi/2,         0,  -np.pi/2,   np.pi/2,         0) #link twist

    # ==================== PLOT ARM ===================================
    def forward_kinematics(self, theta):
        T = np.eye(4)
        link_matrices = [np.copy(T)]
        for i in range(self.num_links):
            T_i = dh_transform_matrix(theta[i], self.d[i], self.a[i], self.alpha[i])
            T = np.dot(T, T_i)
            link_matrices.append(np.copy(T))
        x, y, z = T[:3, 3]
        roll = np.arctan2(T[2, 1], T[2, 2])
        pitch = np.arctan2(-T[2, 0], np.sqrt(T[2, 1] ** 2 + T[2, 2] ** 2))
        yaw = np.arctan2(T[1, 0], T[0, 0])
        dcm = np.zeros((3,3))

        #1 lấy vector theo cột
        # dcm[0,0] = np.cos(yaw)*np.cos(pitch)
        # dcm[0,1] = np.cos(yaw)*np.sin(pitch)*np.sin(roll) - np.sin(yaw)*np.cos(roll)
        # dcm[0,2] = np.cos(yaw)*np.sin(pitch)*np.cos(roll) + np.sin(yaw)*np.sin(roll)
        # dcm[1,0] = np.sin(yaw)*np.cos(pitch)
        # dcm[1,1] = np.sin(yaw)*np.sin(pitch)*np.sin(roll) + np.cos(yaw)*np.cos(roll)
        # dcm[1,2] = np.sin(yaw)*np.sin(pitch)*np.cos(roll) - np.cos(yaw)*np.sin(roll)
        # dcm[2,0] = -np.sin(pitch)
        # dcm[2,1] = np.cos(pitch)*np.sin(roll)
        # dcm[2,2] = np.cos(pitch)*np.cos(roll)

        #2 lấy vector theo hàng
        dcm[0,0] = np.cos(pitch)*np.cos(yaw)
        dcm[0,1] = np.cos(pitch)*np.sin(yaw)
        dcm[0,2] = -np.sin(pitch)
        dcm[1,0] = np.sin(roll)*np.sin(pitch)*np.cos(yaw) - np.cos(roll)*np.sin(yaw)       
        dcm[1,1] = np.sin(roll)*np.sin(pitch)*np.sin(yaw) + np.cos(roll)*np.cos(yaw)
        dcm[1,2] = np.sin(roll)*np.cos(pitch)
        dcm[2,0] = np.cos(yaw)*np.sin(pitch)*np.cos(roll) + np.sin(yaw)*np.sin(roll)
        dcm[2,1] = np.cos(roll)*np.sin(pitch)*np.sin(yaw) - np.sin(roll)*np.cos(yaw)
        dcm[2,2] = np.cos(pitch)*np.cos(roll)
        
        n_x = np.cos(pitch)*np.cos(yaw)
        n_y = np.cos(pitch)*np.sin(yaw)
        n_z = -np.sin(pitch)
        
        end_effector = np.hstack((x, y, z, n_x, n_y, n_z))
        return end_effector, link_matrices, dcm
    
    def plot_robotic_arm(self, theta, ax):
        end_effector, link_matrices, dcm = self.forward_kinematics(theta)
        # Plotting
        ax.set_xlim([-self.max_range, self.max_range])  # Adjust based on your arm dimensions
        ax.set_ylim([-self.max_range, self.max_range])  # Adjust based on your arm dimensions
        ax.set_zlim([0, self.max_range])
        ax.set_box_aspect([1, 1, 1])
        # Plot each link separately
        for i in range(0, self.num_links):
            x_vals = [link_matrices[i][0, 3], link_matrices[i+1][0, 3]]
            y_vals = [link_matrices[i][1, 3], link_matrices[i+1][1, 3]]
            z_vals = [link_matrices[i][2, 3], link_matrices[i+1][2, 3]]
            ax.plot(x_vals, y_vals, z_vals, linewidth=1) #, label=f'Link {i+1}')
        # Plot end-effector
        # ax.scatter([link_matrices[-1][0, 3]], [link_matrices[-1][1, 3]], [link_matrices[-1][2, 3]], color='k', marker='o', s=10, label='End-Effector')
        # dcm=dcm.T nếu dùng 1
        # print(np.linalg.norm(dcm[0,:]))
        # ax.quiver([link_matrices[-1][0, 3]], [link_matrices[-1][1, 3]], [link_matrices[-1][2, 3]], dcm[0,0]*100, dcm[0,1]*100, dcm[0,2]*100, color='red', label='vectorX')
        # ax.quiver([link_matrices[-1][0, 3]], [link_matrices[-1][1, 3]], [link_matrices[-1][2, 3]], dcm[1,0]*100, dcm[1,1]*100, dcm[1,2]*100, color='green', label='vectorY')
        # ax.quiver([link_matrices[-1][0, 3]], [link_matrices[-1][1, 3]], [link_matrices[-1][2, 3]], dcm[2,0]*100, dcm[2,1]*100, dcm[2,2]*100, color='blue', label='vectorZ')
        # ax.quiver([link_matrices[-1][0, 3]], [link_matrices[-1][1, 3]], [link_matrices[-1][2, 3]], end_effector[3]*100, end_effector[4]*100, end_effector[5]*100, color='red', label='vectorX')
    # ====================== GENERATE DATA ==========================    
    
    def batch_forward_kinematics(self, theta):
        T = np.tile(np.eye(4), (theta.shape[0], 1, 1))
        joint_positions = []
        for i in range(self.num_links):
            T_i = batch_dh_transform_matrix(theta[:, i], self.d[i], self.a[i], self.alpha[i])
            T = np.einsum('...ij,...jk->...ik', T, T_i)
            joint_positions.append(T[:, :3, 3].T)
        x, y, z = T[:, :3, 3].T
        # roll = np.arctan2(T[:, 2, 1], T[:, 2, 2])
        pitch = np.arctan2(-T[:, 2, 0], np.sqrt(T[:, 2, 1] ** 2 + T[:, 2, 2] ** 2))
        yaw = np.arctan2(T[:, 1, 0], T[:, 0, 0])
        n_x = np.cos(pitch)*np.cos(yaw)
        n_y = np.cos(pitch)*np.sin(yaw)
        n_z = -np.sin(pitch)
        end_effector = np.column_stack((x, y, z, n_x, n_y, n_z))
        joint_positions = np.transpose(np.array(joint_positions), (2, 0, 1))
        # end_effector = np.column_stack((x, y, z, roll, pitch, yaw))
        return end_effector, joint_positions

    def create_random_data(self, batch_size):
        theta = np.random.uniform(low=self.q_low, high=self.q_high, size=(2*batch_size, self.num_links))
        end_effector, _ = self.batch_forward_kinematics(theta)
        # keep only rows where z >= 0
        # z_limitation = (end_effector[:, 2] >= 0) & (end_effector[:, 2] <= 100)
        # x_limitation = (end_effector[:, 0] >= 400) & (end_effector[:, 0] <= 600)
        # y_limitation = (end_effector[:, 1] >= -400) & (end_effector[:, 1] <= 400)
        # valid_indices = np.logical_and.reduce((z_limitation, x_limitation, y_limitation))
        valid_indices = end_effector[:, 2] >= 0
        end_effector = end_effector[valid_indices]
        theta = theta[valid_indices]
        return end_effector, theta

    def generate_random_data(self, batch_size=1024):
        while True:
            self.end_effector, self.theta = self.create_random_data(batch_size)
            while self.theta.shape[0] < batch_size:
                end_effector, theta = self.create_random_data(batch_size)
                self.end_effector = np.vstack([self.end_effector, end_effector])
                self.theta = np.vstack([self.theta, theta])
            yield self.end_effector[:batch_size], self.theta[:batch_size]
    
    # ====================== CUSTOM LOSS ==========================    
    @tf.function
    def tf_batch_forward_kinematics(self, theta):
        T = tf.eye(4, batch_shape=[tf.shape(theta)[0]], dtype=tf.float32)
        joint_positions = []
        for i in range(self.num_links):
            T_i = tf_batch_dh_transform_matrix(theta[:, i], self.d[i], self.a[i], self.alpha[i])
            T = tf.einsum('...ij,...jk->...ik', T, T_i)
            joint_positions.append(T[:, :3, 3])
        x, y, z = tf.unstack(T[:, :3, 3], axis=1)
        # roll = tf.atan2(T[:, 2, 1], T[:, 2, 2])
        pitch = tf.atan2(-T[:, 2, 0], tf.sqrt(T[:, 2, 1] ** 2 + T[:, 2, 2] ** 2))
        yaw = tf.atan2(T[:, 1, 0], T[:, 0, 0])
        n_x = tf.cos(pitch)*tf.cos(yaw)
        n_y = tf.cos(pitch)*tf.sin(yaw)
        n_z = -tf.sin(pitch)
        end_effector = tf.stack([x, y, z, n_x, n_y, n_z], axis=1)
        joint_positions = np.stack(joint_positions, axis=2)
        return end_effector, joint_positions

    @tf.function
    def get_physics_loss(self, y_true, y_pred):
        # end_effector_true, _ = self.tf_batch_forward_kinematics(y_true)
        # end_effector_pred, _ = self.tf_batch_forward_kinematics(y_pred)
        position_loss = self.get_position_loss(y_true, y_pred)
        orientation_loss = self.get_orientation_loss(y_true, y_pred)
        return position_loss + orientation_loss

    @tf.function
    def get_position_loss(self, y_true, y_pred):
        end_effector_true, _ = self.tf_batch_forward_kinematics(y_true)
        end_effector_pred, _ = self.tf_batch_forward_kinematics(y_pred)
        return tf.reduce_mean(tf.square(end_effector_true[:, :3] - end_effector_pred[:, :3]))
    
    @tf.function
    def get_orientation_loss(self, y_true, y_pred):
        end_effector_true, _ = self.tf_batch_forward_kinematics(y_true)
        end_effector_pred, _ = self.tf_batch_forward_kinematics(y_pred)
        return tf.reduce_mean(tf.square(end_effector_true[:, 3:] - end_effector_pred[:, 3:]))

    @tf.function
    def get_BC_loss(self, y_true, y_pred):
        angle_limit_loss = self.get_angle_limitation_loss(y_pred)
        ground_avoidance_loss = self.get_ground_avoidance_loss(y_pred)
        return angle_limit_loss + ground_avoidance_loss
    
    @tf.function
    def get_angle_limitation_loss(self, y_pred):
        loss_per_joint = tf.reduce_mean(tf.square(
            tf.maximum(self.q_low - y_pred, 0.0)
            + tf.maximum(y_pred - self.q_high, 0.0)
            ), axis=0
        )
        return tf.reduce_mean(loss_per_joint)
    
    @tf.function
    def get_ground_avoidance_loss(self, y_pred):
        _, joint_positions = self.tf_batch_forward_kinematics(y_pred)
        z_values = joint_positions[:, 2, :-1]
        return tf.reduce_mean(tf.square(tf.maximum(self.radius - z_values, 0.0)))

    @tf.function
    def get_total_loss(self, y_true, y_pred):
        position_loss = tf.cast(self.get_position_loss(y_true, y_pred), tf.float32)
        orientation_loss = tf.cast(self.get_orientation_loss(y_true, y_pred), tf.float32)
        BC_loss = tf.cast(self.get_BC_loss(y_true, y_pred), tf.float32)
        return position_loss + orientation_loss + BC_loss
    
    # ====================== EVALUATE MODEL ==========================