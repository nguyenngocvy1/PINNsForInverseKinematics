from math import *
import os
from pathlib import Path

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras_tuner.tuners import BayesianOptimization, Hyperband
from keras import regularizers


class Tuner:
    def __init__(
        self, arm, tuner_dir="tuner", model_name="Untitled_model", algorithm=None
    ):
        self.arm = arm
        self.epochs = 100
        self.steps_per_epoch = 1000
        self.validation_steps = self.steps_per_epoch // 4
        self.batch_size = 4096
        self.train_generator = arm.generate_random_data(self.batch_size)
        self.test_generator = arm.generate_random_data(self.batch_size)
        self.num_trials = 20
        self.tuner_dir = tuner_dir
        if not os.path.exists(model_name):
            os.mkdir(model_name)
        self.folder = Path(model_name)
        self.algorithm = algorithm
        self.X_val, self.y_val = next(arm.generate_random_data(1))
        self.num_input_features = self.X_val.shape[1]
        self.num_output_features = self.y_val.shape[1]
        self.num_layers = 5
        self.num_neurons = 1024
        self.activations = ["relu", "leaky_relu"]
        self.learning_rates = [1e-4, 1e-5, 1e-6, 1e-7]
        self.best_model, self.best_hyperparameters = self.tune_model()
        self.write_best_model()

    def build_model(self, hp):  # Define the Keras Tuner search space
        model = Sequential()
        model.add(
            Dense(
                units=hp.Int(
                    "units_1", min_value=32, max_value= 2*self.num_neurons, step=32
                ),
                input_shape=(self.num_input_features,),
                activation=hp.Choice("activation_1", values=self.activations),
                kernel_regularizer=regularizers.l2(0.01),
            )
        )
        model.add(
                Dropout(
                    rate=hp.Float(
                        f"dropout_1", min_value=0.0, max_value=0.5, step=0.1
                    )
                )
            )
        model.add(Dense(self.num_output_features, activation="linear"))
        optimizer = Adam(
            learning_rate=hp.Choice("learning_rate", values=self.learning_rates),
        )
        for i in range(self.num_layers):
            model.add(
                Dense(
                    units=hp.Int(
                        f"units_{i+2}",
                        min_value=32,
                        max_value=self.num_neurons,
                        step=32,
                    ),
                    activation=hp.Choice(f"activation_{i+2}", values=self.activations),
                    kernel_regularizer=regularizers.l2(0.01),
                )
            )
            model.add(
                Dropout(
                    rate=hp.Float(
                        f"dropout_{i+2}", min_value=0.0, max_value=0.5, step=0.1
                    )
                )
            )
        model.add(Dense(self.num_output_features, activation="linear"))
        optimizer = Adam(
            learning_rate=hp.Choice("learning_rate", values=self.learning_rates),
        )
        model.compile(
            optimizer=optimizer,
            loss=self.arm.get_total_loss,
            metrics=[
                self.arm.get_physics_loss,
                self.arm.get_BC_loss,
            ],
        )
        return model

    def tune_model(self):
        if "Bayesian" in str(self.algorithm):
            tuner = BayesianOptimization(
                self.build_model,
                objective="val_loss",
                max_trials=self.num_trials,
                directory=self.folder / self.tuner_dir,
                project_name="hyperparameter_tuning",
            )
        else:
            tuner = Hyperband(
                self.build_model,
                objective="val_loss",
                max_epochs=self.epochs,
                factor=3,
                directory=self.folder / self.tuner_dir,
                project_name="hyperparameter_tuning",
            )
        tuner.search(
            self.train_generator,
            validation_data=self.test_generator,
            epochs=self.epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            batch_size=self.batch_size,
            callbacks=[
                EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=3)
            ],
        )
        best_model = tuner.get_best_models(num_models=1)[0]
        best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]
        return best_model, best_hyperparameters

    def write_best_model(self):
        with open(self.folder / "model.txt", "w") as file:
            file.write("Best Hyperparameters:\n")
            file.write(str(self.best_hyperparameters.get_config()))
            print("Best Model Summary:", file=file)
            self.best_model.summary(print_fn=lambda x: file.write(x + "\n"))
            file.write("\n\nLearning Rate:\n")
            file.write(str(self.best_hyperparameters.get("learning_rate")))
            file.write("\n\nActivation:\n")
            activation_hyperparams = {
                key: value
                for key, value in self.best_hyperparameters.values.items()
                if "activation" in key
            }
            for key, value in activation_hyperparams.items():
                file.write(f"{key}: {value}\n")
            file.write("\nDropout:\n")
            dropout_hyperparams = {
                key: value
                for key, value in self.best_hyperparameters.values.items()
                if "dropout" in key
            }
            for key, value in dropout_hyperparams.items():
                file.write(f"{key}: {value}\n")
            file.write("Regularization:\n")
            regularization_hyperparams = {
                key: value
                for key, value in self.best_hyperparameters.values.items()
                if "regularization" in key
            }
            for key, value in regularization_hyperparams.items():
                file.write(f"{key}: {value}\n")
