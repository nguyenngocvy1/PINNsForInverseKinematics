import sys
import time
from threading import Thread

import serial
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QLineEdit
from keras.models import load_model
from keras.utils import get_custom_objects
import numpy as np

from Arm3D.config import RoboticsArm

def get_line_trajectory(initial_point=[300, -200, 200], final_point=[300, 200, 200], num_points=90):
    # Initial and final points
    initial_point = np.array(initial_point)
    final_point = np.array(final_point)
    # Generate straight line trajectory
    xyz = np.linspace(initial_point, final_point, num_points)
    vector0 = np.zeros((num_points, 2))    
    vector1 = np.ones((num_points, 1))
    return np.hstack((xyz, vector0, -vector1))

def get_circular_trajectory(center=[300, 0, 200], radius=50.0, num_points=90):
# Center and radius of the circle
    center = np.array(center)
    # Angle parameter for the circle
    theta = np.linspace(0, 2 * np.pi, num_points)
    # Generate circular trajectory
    x_circle = center[0] + radius * np.cos(theta)
    y_circle = center[1] + radius * np.sin(theta)
    z_circle = center[2] + np.zeros_like(theta)  # Z coordinate remains constant
    vector0 = np.zeros((num_points, 1))
    vector1 = np.ones((num_points, 1))
    return np.column_stack((x_circle, y_circle, z_circle, vector0, vector0, -vector1))

def get_helix_trajectory(center=[300, 0, 100], radius=100.0, num_points=90):
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

class RobotControlGUI(QWidget):
    def __init__(self):
        super().__init__()

        self.init_ui()
    
    def init_ui(self):
        # Create widgets
        self.setWindowTitle('Robot Control GUI')
        self.move_to_button = QPushButton('Move To', self)
        self.homing_button = QPushButton('Homing', self)
        self.get_angles_button = QPushButton('Get Current Angles', self)
        self.connect_button = QPushButton('Connect', self)
        self.disconnect_button = QPushButton('Disconnect', self)
        self.line_button = QPushButton('Line', self)
        self.circle_button = QPushButton('Circle', self)
        self.output_label = QLabel('Output will be displayed here.', self)
        self.waypoint_label = QLabel('Input waypoint here.', self)
        self.joint_label = QLabel('Input joint here.', self)
        self.com_label = QLabel('Input Port here.', self)
        self.waypoint_input = QLineEdit(self)
        self.joint_input = QLineEdit(self)
        self.com_input = QLineEdit(self)
        
        # Set placeholder text
        self.com_input.setPlaceholderText('COM8')
        self.waypoint_input.setPlaceholderText('x y z nx ny nz')
        self.joint_input.setPlaceholderText('1')
        
        # Set up layout
        layout = QVBoxLayout(self)
        layout.addWidget(self.com_label)
        layout.addWidget(self.com_input)
        layout.addWidget(self.connect_button)
        layout.addWidget(self.disconnect_button)
        layout.addWidget(self.waypoint_label)
        layout.addWidget(self.waypoint_input)
        layout.addWidget(self.move_to_button)
        layout.addWidget(self.line_button)
        layout.addWidget(self.circle_button)
        layout.addWidget(self.joint_label)
        layout.addWidget(self.joint_input)
        layout.addWidget(self.homing_button)
        layout.addWidget(self.get_angles_button)
        layout.addWidget(self.output_label)
        
        # Connect buttons to functions
        self.connect_button.clicked.connect(self.connect)
        self.disconnect_button.clicked.connect(self.disconnect)
        self.move_to_button.clicked.connect(self.move_to)
        self.homing_button.clicked.connect(self.homing)
        self.get_angles_button.clicked.connect(self.get_current_angles)
        self.line_button.clicked.connect(self.line)
        self.circle_button.clicked.connect(self.circle)
        
        # Variable
        self.data_serial = None
        self.joint_list = {'X', 'Y', 'Z', 'A', 'B'}
        self.count = 0
        self.response = ''
    
    # Communication
    def connect(self):
        port = self.com_input.text() or 'COM8'
        try:
            self.data_serial = serial.Serial(port, 250000)
            print('Serial port opened')
        except Exception as e:
            print(f'Error: {e}')
        
    def disconnect(self):
        self.data_serial.close()
        self.count = 0
        print('Serial port closed')

    
    def send_serial(self, gcode):
        try:
            self.data_serial.write((gcode+'\n').encode("utf-8"))
            print(gcode + '\n')
            # if "G0 " in gcode:
            #     response = ''
            #     while True:
            #         if 'ok' in response:
            #             break
            #         response = self.data_serial.readline().decode("utf-8")
            # else:
            time.sleep(0.1)
            response = self.data_serial.readline().decode("utf-8")
            print(response)
            return response
        except Exception as e:
            print(f'Error: {e}')
    
    # Command    
    def move_to(self):
        waypoint = [[float(value) for value in self.waypoint_input.text().split()]]
        return self.get_model(waypoint)
    
    def get_model(self, points):
        predictions = model.predict(points)
        predicted_angles = np.round(self.convert_angles(np.degrees(predictions)), 2)
        self.rotate_arm(predicted_angles)
        return predicted_angles
    
    # def rotate_angles(self):
    #     angles = [[float(value) for value in self.waypoint_input.text().split()]]
    #     return self.rotate_arm(angles)
                    
    def rotate_arm(self, predicted_angles):
        command_strings = ["G0 X"+str(angles[0])+ " Y"+str(angles[1])+" Z"+str(angles[2])+" A"+str(angles[3])+" B"+str(angles[4])+" F500" for angles in predicted_angles]
        # if len(command_strings) >= 5:
            # command_strings = ['\n'.join(command_strings[i:i+5]) for i in range(0, len(command_strings), 5)]
        # command_strings = '\n'.join(command_strings)
        # self.send_serial(command_string)
        for command in command_strings:
            response = self.send_serial(command)
        else:
            print('finish')
        return None

    def convert_angles(self, angles):
        converted_angles = np.zeros_like(angles)
        converted_angles[:, 0] = angles[:, 0]
        converted_angles[:, 1] = 90 - angles[:, 1]
        converted_angles[:, 2] = -90 - angles[:, 2]
        converted_angles[:, 3] = -angles[:, 3]
        converted_angles[:, 4] = 90 - angles[:, 4]
        return converted_angles
    
    def homing(self):
        self.count += 1
        joint_code = self.convert_joint()
        if joint_code is None:
            if self.count > 1:
                self.rotate_arm([[10, 10, 10, 10, 10]])
            self.send_serial("G28")
            return self.rotate_arm([[0, 0, 0, 0, 0]])
        else:
            joint = int(self.joint_input.text())-1
            current_angles = self.get_current_angles()
            if current_angles[joint] < 0:
                rotate_angles = current_angles.copy
                rotate_angles[joint] = 10
                self.rotate_arm(rotate_angles)
            return self.send_serial(f"G28 {joint_code}")

    def get_current_angles(self):
        pass
        joint_code = self.convert_joint()
        if joint_code is None:
            angles = []
            response = self.send_serial("M114")
            start_index = response.find("X:")
            end_index = response.find("Count")
            input_string = response[start_index:end_index].strip()
            angles_string = input_string.split()
            for angle_string in angles_string:
                try:
                    _, angle = angle_string.split(':')
                except:
                    pass
                angles.append(float(angle))
            if len(angles)==5:
                self.output_label.setText(f'Current Angles: {angles}')
                return angles
            return None
        else:
            return self.send_serial(f"M114 {joint_code}")
    
    def convert_joint(self):
        try: 
            return self.joint_list[int(self.joint_input.text())-1]
        except:
            return None
    
    #trajectory
    def line(self):
        line_points = get_line_trajectory()
        self.get_model(line_points)
        

    def circle(self):
        circle_points = get_circular_trajectory()
        self.get_model(circle_points)
         
            

if __name__ == "__main__":
    try:
        get_custom_objects().update({"get_total_loss": RoboticsArm.get_total_loss})
        get_custom_objects().update({"get_physics_loss": RoboticsArm.get_physics_loss})
        get_custom_objects().update({"get_BC_loss": RoboticsArm.get_BC_loss})
        get_custom_objects().update({"get_position_loss": RoboticsArm.get_position_loss})
        get_custom_objects().update({"get_orientation_loss": RoboticsArm.get_orientation_loss})
        model = load_model("C:\\MyFolder\\Code\\Python\\Thesis\\final\\model_12_29_14_42_[0.13155946135520935, 0.18043625354766846, 4.208344398648478e-05]\\model.h5")
        app = QApplication(sys.argv)
        window = RobotControlGUI()
        window.show()
        sys.exit(app.exec_())
    except Exception as e:
        print(f'Error: {e}')

    
    
    
        


    
    
    