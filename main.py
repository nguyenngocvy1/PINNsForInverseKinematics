import numpy as np
import datetime
import os
import logging

from Arm3D.config import RoboticsArm
from Arm3D.tuner import Tuner
from Arm3D.models import PINNs

def train(arm):
    my_list = ['Hyperband', 'BayesianOptimization']
    now = datetime.datetime.now()
    for tuner_algorithm in my_list:
        model_name = f"model_{tuner_algorithm}_{now.month}_{now.day}_{now.hour}_{now.minute}"
        if not os.path.exists(model_name):
            os.mkdir(model_name)
        tuner  = Tuner(arm, tuner_dir='my_dir', model_name=model_name, algorithm=tuner_algorithm)
        trained_model = PINNs(arm, model=tuner.best_model, model_name=model_name)

def train_without_tuning(arm):
    now = datetime.datetime.now()
    model_name = f"model_{now.month}_{now.day}_{now.hour}_{now.minute}"
    if not os.path.exists(model_name):
        os.mkdir(model_name)
    trained_model = PINNs(arm, model_name=model_name)
    
def main():
    logging.basicConfig(filename='log.txt', level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    arm_5DOF = RoboticsArm(
        lengths=(231.5, 221.12, 0, 223, 170),
        q_low=np.radians([-60, 0, -180, -45, 0]),
        q_high=np.radians([60, 90, -90, 45, 90])
        )
    choice = input("tuning or not?")
    if "no" in choice:
        train_without_tuning(arm_5DOF)
    else:      
        train(arm_5DOF)
    
if __name__ == "__main__":
    main()