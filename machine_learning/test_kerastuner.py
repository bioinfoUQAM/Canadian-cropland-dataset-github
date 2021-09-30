from tensorflow import keras
import numpy as np

(x, y), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x[:-10000]
x_val = x[-10000:]
y_train = y[:-10000]
y_val = y[-10000:]

x_train = np.expand_dims(x_train, -1).astype("float32") / 255.0
x_val = np.expand_dims(x_val, -1).astype("float32") / 255.0
x_test = np.expand_dims(x_test, -1).astype("float32") / 255.0

num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_val = keras.utils.to_categorical(y_val, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

"""
## Prepare a model-building function

Then, we define a model-building function. It takes an argument `hp` from
which you can sample hyperparameters, such as
`hp.Int('units', min_value=32, max_value=512, step=32)`
(an integer from a certain range).

This function returns a compiled model.
"""

from tensorflow.keras import layers
from kerastuner import RandomSearch


def build_model(hp):
    model = keras.Sequential()
    model.add(layers.Flatten())
    model.add(
        layers.Dense(
            units=hp.Int("units", min_value=32, max_value=512, step=32),
            activation="relu",
        )
    )
    model.add(layers.Dense(10, activation="softmax"))
    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])
        ),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


"""
## Start the search

Next, let's instantiate a tuner. You should specify the model-building function, the
name of the objective to optimize (whether to minimize or maximize is
automatically inferred for built-in metrics), the total number of trials
(`max_trials`) to test, and the number of models that should be built and fit
for each trial (`executions_per_trial`).

We use the `overwrite` argument to control whether to overwrite the previous
results in the same directory or resume the previous search instead.  Here we
set `overwrite=True` to start a new search and ignore any previous results.

Available tuners are `RandomSearch`, `BayesianOptimization` and `Hyperband`.

**Note:** the purpose of having multiple executions per trial is to reduce
results variance and therefore be able to more accurately assess the
performance of a model. If you want to get results faster, you could set
`executions_per_trial=1` (single round of training for each model
configuration).
"""

tuner = RandomSearch(
    build_model,
    objective="val_accuracy",
    max_trials=3,
    executions_per_trial=2,
    overwrite=True,
    directory="my_dir",
    project_name="helloworld",
)

"""
You can print a summary of the search space:
"""

tuner.search_space_summary()

"""
Then, start the search for the best hyperparameter configuration.
The call to `search` has the same signature as `model.fit()`.
"""

tuner.search(x_train, y_train, epochs=2, validation_data=(x_val, y_val))

"""
Here's what happens in `search`: models are built iteratively by calling the
model-building function, which populates the hyperparameter space (search
space) tracked by the `hp` object. The tuner progressively explores the space,
recording metrics for each configuration.
"""

"""
## Query the results

When search is over, you can retrieve the best model(s):
"""

models = tuner.get_best_models(num_models=2)

"""
Or print a summary of the results:
"""

tuner.results_summary()

