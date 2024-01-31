import tensorflow as tf
from tensorflow import keras as keras
import numpy as np


# dynamicly return optimizer with different base
# SGD for shakespeare, and Adam for others
def Scaffold(base_optimizer, learning_rate=None, **kwargs):

    if base_optimizer == "SGD":
        base = keras.optimizers.SGD
    else:
        base = keras.optimizers.Adam

    class Scaffold(base):
        def __init__(
            self,
            learning_rate=0.001,
            **kwargs,
        ):
            super(Scaffold, self).__init__(
                name="scaffold", learning_rate=learning_rate, **kwargs
            )

        def _get_gradients(self, tape, loss, var_list, grad_loss=None):
            grads = tape.gradient(loss, var_list, grad_loss)

            # c_diff = - ci + c
            grads = [
                grads_layer + c_diff_layer
                for grads_layer, c_diff_layer in zip(grads, self.c_diff)
            ]
            return list(zip(grads, var_list))

        def set_controls(self, weights, st=None):
            server_controls = st.server_controls if st else None
            local_controls = st.local_controls if st else None

            # c:  server controls
            # ci: client controls (local)
            # c_diff: (-ci + c) = c - ci
            self.c = (
                tf.nest.map_structure(
                    lambda array: tf.Variable(array, dtype=tf.float32), server_controls
                )
                if server_controls
                else [
                    tf.Variable(tf.zeros(shape=layer.shape, dtype=tf.float32))
                    for layer in weights
                ]
            )
            self.ci = (
                tf.nest.map_structure(
                    lambda array: tf.Variable(array, dtype=tf.float32), local_controls
                )
                if local_controls
                else [
                    tf.Variable(tf.zeros(shape=layer.shape, dtype=tf.float32))
                    for layer in weights
                ]
            )

            # c_diff = -ci + c = c - ci
            self.c_diff = [
                tf.Variable(tf.subtract(c_layer, ci_layer))
                for c_layer, ci_layer in zip(self.c, self.ci)
            ]

        def get_new_client_controls(self, global_weights, local_weights, option=1):
            # model difference (global - local)
            model_diff = [
                np.subtract(global_layer, local_layer)
                for global_layer, local_layer in zip(global_weights, local_weights)
            ]

            if option == 1:
                return model_diff
            else:
                local_lr = float(self.lr)
                local_steps = int(self.iterations.value())

                scale = 1 / (local_steps * local_lr)
                ci_new = [
                    # local_control - server_control + scale * delta
                    np.add(
                        np.subtract(local_control, server_control),
                        np.multiply(scale, delta),
                    )
                    for local_control, server_control, delta in zip(
                        self.ci, self.c, model_diff
                    )
                ]
                return ci_new

        def get_config(self):
            config = super().get_config()
            # config.update()
            return config

    return (
        Scaffold(learning_rate=learning_rate, **kwargs)
        if learning_rate is not None
        else Scaffold(**kwargs)
    )
