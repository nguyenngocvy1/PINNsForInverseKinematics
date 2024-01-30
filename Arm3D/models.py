from math import *
import os
from pathlib import Path
import pickle

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras import regularizers
import matplotlib.pyplot as plt

from .loss_balancer import LossBalancer


class PINNs:
    def __init__(self, arm, model=None, model_name="Untitled_model"):
        self.arm = arm
        self.model_name = model_name
        self.patience = 10  # if the validation loss does not improve for 10 epochs, the training will be stopped
        self.epochs = 500
        self.steps_per_epoch = 10_000
        self.validation_steps = self.steps_per_epoch // 4
        self.batch_size = 4096
        self.train_generator = arm.generate_random_data(self.batch_size)
        self.test_generator = arm.generate_random_data(self.batch_size)
        self.X_val, self.y_val = next(arm.generate_random_data(self.steps_per_epoch))
        self.num_input_features = self.X_val.shape[1]
        self.num_output_features = self.y_val.shape[1]
        self.my_model = model or self.get_neural_network()
        self.history = self.train_neural_network()
        self.plot_loss()
        self.plot_metrics()

    def get_neural_network(self):
        # Define the ANN architecture
        model = Sequential()
        model.add(
            Dense(128, input_shape=(self.num_input_features,), activation="leaky_relu", kernel_regularizer=regularizers.l2(0.01))
        )
        model.add(Dropout(0.2))
        model.add(
            Dense(64, activation="leaky_relu", kernel_regularizer=regularizers.l2(0.01))
        )
        model.add(Dropout(0.2))
        model.add(
            Dense(64, activation="leaky_relu", kernel_regularizer=regularizers.l2(0.01))
        )
        model.add(
            Dense(64, activation="leaky_relu", kernel_regularizer=regularizers.l2(0.01))
        )
        model.add(Dropout(0.2))
        model.add(Dense(self.num_output_features, activation="linear"))
        model.compile(
            optimizer=Adam(learning_rate=1e-5),
            loss=[self.arm.get_position_loss, self.arm.get_orientation_loss, self.arm.get_BC_loss],
            loss_weights=[0.3, 0.3, 0.4],
            metrics=[self.arm.get_position_loss, self.arm.get_orientation_loss, self.arm.get_BC_loss],
        )
        return model

    def train_neural_network(self):
        history = self.my_model.fit(
            self.train_generator,
            validation_data=self.test_generator,
            epochs=self.epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            batch_size=self.batch_size,
            callbacks=[
                LossBalancer(alpha=0.1, lookback_window=10),
                EarlyStopping(
                    monitor="val_loss", mode="min", verbose=1, patience=self.patience
                )
            ],
        )
        self.loss = self.valid_neural_network()
        self.folder_name = f"{self.model_name}_{self.loss}"
        os.rename(self.model_name, f"{self.model_name}_{self.loss}")
        self.folder = Path(self.folder_name)
        self.my_model.save(self.folder / "model.h5")
        with open(self.folder / "history.pickle", "wb") as file:
            pickle.dump(history.history, file)
        return history.history

    def valid_neural_network(self):
        return self.my_model.evaluate(self.X_val, self.y_val)

    def plot_loss(self):
        plt.figure()
        plt.plot(self.history["loss"])
        plt.plot(self.history["val_loss"])
        plt.title("Model loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend(["Train", "Validation"], loc="upper right")
        plt.savefig(self.folder / "train_test.png")

    def plot_metrics(self):
        plt.figure()
        position_loss_values = self.history["get_position_loss"]
        orientation_loss_values = self.history["get_orientation_loss"]
        BC_loss_values = self.history["get_BC_loss"]
        epochs = range(1, len(position_loss_values) + 1)
        plt.plot(epochs, position_loss_values, "r", label="Position Loss")
        plt.plot(epochs, orientation_loss_values, "g", label="Orienation Loss")
        plt.plot(epochs, BC_loss_values, "b", label="BC Loss")
        plt.title("Training Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(self.folder / "metrics.png")
