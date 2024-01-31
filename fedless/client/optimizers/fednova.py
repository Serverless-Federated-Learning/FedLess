import tensorflow as tf
from tensorflow import keras as keras


class FedNova(keras.optimizers.SGD):
    def __init__(
        self,
        mu=0.0,
        learning_rate=0.001,
        momentum=0.9,
        **kwargs,
    ):
        super(FedNova, self).__init__(
            name="fednova", learning_rate=learning_rate, momentum=momentum, **kwargs
        )
        self.mu = tf.Variable(mu)
        self.local_normalizing_vec = tf.Variable(0.0)
        self.local_counter = tf.Variable(0.0)
        self.local_steps = tf.Variable(0)

        # global model weights
        self.global_weights = None
        self.acc_grads = None

    @tf.function
    def compute_local_norm(self):
        var_dtype = self.mu.dtype
        lr_t = self.lr
        momentum_t = self.momentum
        # lr_t = self._get_hyper("learning_rate", var_dtype)
        # momentum_t = self._get_hyper("momentum", var_dtype)

        # compute local normalizing vector a_i
        if momentum_t != 0:
            self.local_counter.assign(self.local_counter * momentum_t + 1)
            self.local_normalizing_vec.assign_add(self.local_counter)

        etamu = lr_t * self.mu
        if etamu != 0:
            self.local_normalizing_vec.assign(self.local_normalizing_vec * (1 - etamu))
            self.local_normalizing_vec.assign_add(1)

        if momentum_t == 0 and etamu == 0:
            self.local_normalizing_vec.assign_add(1)

        self.local_steps.assign_add(1)

    def _get_gradients(self, tape, loss, var_list, grad_loss=None):
        grads = tape.gradient(loss, var_list, grad_loss)

        # cum grad
        for acc_grad, grad in zip(self.acc_grads, grads):
            acc_grad.assign_add(grad * self.lr)

        # apply proximal updates
        grads = [
            grads_layer + self.mu * (local_layer - global_layer)
            for grads_layer, local_layer, global_layer in zip(
                grads, var_list, self.global_weights
            )
        ]
        return list(zip(grads, var_list))

    def minimize(self, loss, var_list, grad_loss=None, name=None, tape=None):
        self.compute_local_norm()

        grads_and_vars = self._compute_gradients(
            loss, var_list=var_list, grad_loss=grad_loss, tape=tape
        )
        return self.apply_gradients(grads_and_vars, name=name)

    def set_global_weights(self, weights):
        self.global_weights = weights
        self.acc_grads = [
            tf.Variable(tf.zeros(shape=layer.shape, dtype=tf.float32))
            for layer in weights
        ]

    def get_acc_grads(self):
        return [layer.numpy() for layer in self.acc_grads]

    def get_local_counters(self):
        # convert to native type for serialization
        return {
            "local_steps": int(self.local_steps.numpy()),
            "local_counter": float(self.local_counter.numpy()),
            "local_normalizing_vec": float(self.local_normalizing_vec.numpy()),
        }

    def get_config(self):
        config = {}  # super().get_config()
        config.update(self.get_local_counters())
        return config
