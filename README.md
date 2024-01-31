# PINNs for Inverse Kinematics in 5DOF BCN3D MOVEO

## Introduction
This project implements Physics-Informed Neural Networks (PINNs) for solving the Inverse Kinematics problem in a 5 Degree of Freedom (5DOF) robotic arm, specifically the BCN3D MOVEO. The goal is to provide an efficient and accurate solution for determining the joint angles required to achieve a desired end-effector position.

## Features
- **PINNs Implementation:** Utilizes Physics-Informed Neural Networks to learn the inverse kinematics of the BCN3D MOVEO robotic arm.
- **Tuning Options:** The project offers the flexibility to choose between different tuning algorithms such as Hyperband and Bayesian Optimization to enhance the model's performance.
- **Logging:** Detailed logs are maintained in 'log.txt' to keep track of training progress and any potential issues.

## Dependencies
- Python 3.x
- Numpy
- Tensorflow
- Keras
- Matplotlib
- PyQt5

## Installation
```
git clone https://github.com/nguyenngocvy1/PINNsForInverseKinematics.git
```

## Install the required dependencies:
```
pip install -r requirements.txt
```

## Project Structure
- **main.py:** The main script to initiate the training process.
- **result.py:** The main script to initiate the Evaluate process.
- **control.py:** The main script to initiate robot control UI.
- **Arm3D Library:** A library containing configurations tuners, and models specific to the BCN3D MOVEO robotic arm.
    - **config.py:** generate data, custom loss
    - **tuner.py:** tune hyperparameter
    - **models.py:** create and train model
## Train model:
```
python main.py
```

## Evaluate model:
1. Run the script using the command:
```
python result.py
```
2. Choose the type of trajectory you want the robotic arm to draw: 'Line', 'Circle', 'Helix', or any other custom trajectory.

3. The script will load a pre-trained PINNs model and generate the trajectory points based on your choice.

4. The animation will be saved as a GIF file (anime_{choice}_5DOF.gif) in the current working directory.
#### Trajectory Options:
- **Line:** A straight-line trajectory from an initial point to a final point.

- **Circle:** A circular trajectory around a specified center with a given radius.

- **Helix:** A helical trajectory combining circular motion with upward movement along the z-axis.

- **Custom Trajectory:** If the input doesn't match 'Line', 'Circle', or 'Helix', the script generates random data for the trajectory.

#### Results:

- The animation GIF (anime_{choice}_5DOF.gif) showcases the movement of the BCN3D MOVEO robotic arm while following the chosen trajectory.

- Distance and orientation losses are calculated and saved in text files (distance_{choice}.txt and orientation_{choice}.txt, respectively).

- The robotic arm's joint positions and end-effector positions are visualized in the animation.

### Control Arm:
1. Run the script using the command:
```
python control.py
```
2. Enter the COM port (e.g., COM8) and click on the "Connect" button.

3. Click on "Homing" to perform homing, and the robotic arm will move to its home position.

4. Use the "Move To" button to input specific XYZ coordinates and move the robotic arm to that position.

5. Use the "Get Current Angles" button to retrieve the current joint angles of the robotic arm.

6. The "Line" and "Circle" buttons allow you to execute predefined trajectories.

7. Click on the "Disconnect" button to close the serial port connection.

### GUI Layout:

- **Connect/Disconnect:** Enter the COM port and use the "Connect" button to establish a connection. Use the "Disconnect" button to close the connection.

- **Move To:** Input XYZ coordinates in the waypoint input field and click on "Move To" to move the robotic arm to the specified position.

- **Homing:** Click on the "Homing" button to perform homing and move the robotic arm to its home position.

- **Get Current Angles:** Click on this button to retrieve and display the current joint angles of the robotic arm.

- **Line/Circle:** Execute predefined trajectories using the "Line" and "Circle" buttons.

- **Input Fields:** Enter waypoints, joint numbers, or COM port in the corresponding input fields.

- **Output Label:** Displays output information, such as current angles or errors.

## Contact:
For any inquiries or issues, please contact vy070401@gmail.com.