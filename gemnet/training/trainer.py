import tensorflow as tf
import numpy as np
import tensorflow_addons as tfa
from .schedules import LinearWarmupExponentialDecay
import logging


class Trainer:
    """
    Parameters
    ----------
        model: Model
            Model to train.
        learning_rate: float
            Initial learning rate.
        decay_steps: float
            Number of steps until learning rate reaches learning_rate*decay_rate
        decay_rate: float
            Decay rate.
        warmup_steps: int
            Total number of warmup steps of the learning rate schedule..
        weight_decay: bool
            Weight decay factor of the AdamW optimizer.
        staircase: bool
            If True use staircase decay and not (continous) exponential decay
        grad_clip_max: float
            Gradient clipping threshold.
        decay_patience: int
            Learning rate decay on plateau. Number of evaluation intervals after decaying the learning rate.
        decay_factor: float
            Learning rate decay on plateau. Multiply inverse of decay factor by learning rate to obtain new learning rate.
        decay_cooldown: int
            Learning rate decay on plateau. Number of evaluation intervals after which to return to normal operation.
        ema_decay: float
            Decay to use to maintain the moving averages of trained variables.
        rho_force: float
            Weighing factor for the force loss compared to the energy. In range [0,1]
            loss = loss_energy * (1-rho_force) + loss_force * rho_force
        loss: str
            Name of the loss objective of the forces.
        mve: bool
            If True perform Mean Variance Estimation.
        agc: bool
            If True use adaptive gradient clipping else clip by global norm.
    """

    def __init__(
        self,
        model,
        learning_rate: float = 1e-3,
        decay_steps: int = 100000,
        decay_rate: float = 0.96,
        warmup_steps: int = 0,
        weight_decay: float = 0.001,
        staircase: bool = False,
        grad_clip_max: float = 1000,
        decay_patience: int = 10,  # decay lr on plateau by decay_factor
        decay_factor: float = 1,
        decay_cooldown: int = 10,
        ema_decay: float = 0.999,
        rho_force: float = 0.99,
        loss: str = "mae",  # else use rmse
        mve: bool = False,
        agc=False,
    ):
        self.backup_vars = None
        assert 0 <= rho_force <= 1

        self.model = model
        self.ema_decay = ema_decay
        self.grad_clip_max = grad_clip_max
        self.rho_force = tf.constant(rho_force, dtype=tf.float32)
        self.mve = mve
        self.loss = loss
        self.agc = agc
        self.optimizers = {}

        if mve:
            self.tracked_metrics = [
                "loss",
                "energy_mae",
                "energy_nll",
                "energy_var",
                "force_mae",
                "force_rmse",
                "force_nll",
                "force_var",
            ]
        else:
            self.tracked_metrics = ["loss", "energy_mae", "force_mae", "force_rmse"]

        self.reset_optimizer(
            learning_rate,
            warmup_steps,
            decay_steps,
            decay_rate,
            staircase,
            weight_decay,
        )
        self.weight_groups = None
        self.plateau_callback = ReduceLROnPlateau(
            self, decay_patience, decay_factor, cooldown=decay_cooldown
        )

    def _get_variable_copy(self, params):
        return [tf.Variable(var, dtype=var.dtype, trainable=False) for var in params]

    def _group_weights_maybe(self, params):
        """
        Parameters
        ----------
            params: dict
                If None split params into keys 'Adam' and 'AdamW'. All kernels are put into 'AdamW', the rest into 'Adam'.

        Returns
        -------
            params: dict
        """
        if params is None:
            params = {"AdamW": [], "Adam": []}
            for W in self.model.trainable_weights:
                if "kernel" in W.name:
                    params["AdamW"] += [W]
                else:
                    params["Adam"] += [W]
        return params

    def _get_schedule(
        self, learning_rate, warmup_steps, decay_steps, decay_rate, staircase
    ):
        """
        Returns the learning rate scheduler.
        """
        if warmup_steps > 0:
            return LinearWarmupExponentialDecay(
                learning_rate,
                warmup_steps,
                decay_steps,
                decay_rate,
                staircase=staircase,
            )
        else:
            return tf.optimizers.schedules.ExponentialDecay(
                learning_rate, decay_steps, decay_rate, staircase=staircase
            )

    def reset_optimizer(
        self,
        learning_rate,
        warmup_steps,
        decay_steps,
        decay_rate,
        staircase,
        weight_decay,
    ):
        self.learning_rate = self._get_schedule(
            learning_rate, warmup_steps, decay_steps, decay_rate, staircase
        )

        # if not explicitly set optimizer values to a variable then those values will not be tracked by tf
        if weight_decay > 0:
            self.weight_decay = self._get_schedule(
                weight_decay, warmup_steps, decay_steps, decay_rate, staircase
            )

            optW = tfa.optimizers.AdamW(
                weight_decay=self.weight_decay,
                learning_rate=self.learning_rate,
                beta_1=tf.Variable(0.9),
                beta_2=tf.Variable(0.999),
                epsilon=tf.Variable(1e-7),
                amsgrad=True,
            )
        else:
            optW = tf.optimizers.Adam(
                learning_rate=self.learning_rate,
                beta_1=tf.Variable(0.9),
                beta_2=tf.Variable(0.999),
                epsilon=tf.Variable(1e-7),
                amsgrad=True,
            )

        # Optimzer for all weights that are not kernels (kernels = 2/3D weights)
        opt = tf.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=tf.Variable(0.9),
            beta_2=tf.Variable(0.999),
            epsilon=tf.Variable(1e-7),
            amsgrad=True,
        )

        optimizer = tfa.optimizers.MovingAverage(opt, average_decay=self.ema_decay)
        optimizerW = tfa.optimizers.MovingAverage(optW, average_decay=self.ema_decay)

        # bug in tfa < 0.12 (save all optimizers in trainer.optimizers)
        # need to explicitly save the optimizer inside the moving average optimizer
        # if trainer.optimizer._optimizer is not saved then the weights of adam are not saved only those of the moving average optimizer
        self.optimizers["Adam"] = opt
        self.optimizers["AdamW"] = optW
        self.optimizers["AvgAdam"] = optimizer
        self.optimizers["AvgAdamW"] = optimizerW

    def decay_maybe(self, val_loss):
        self.plateau_callback.decay_maybe(val_loss)

    @staticmethod
    def _adaptive_gradient_clipping(lamb, grad, weight, eps=0.001):
        """adapted from High-Performance Large-Scale Image Recognition Without Normalization:
        https://github.com/deepmind/deepmind-research/blob/master/nfnets/optim.py"""
        # cannot clip a gradient that was not calculated
        if grad is None:
            return None

        # no gradient clipping for last linear layer as proposed in the paper
        if "_final" in weight.name:
            return grad

        # clip gradient depending on weight norm
        if len(tf.shape(weight)) > 1:
            axis = 0  # fan-in extent
            if len(tf.shape(weight)) == 3:  # weights of efficient version are 3d
                axis = (0, 1)  # fan-in extent
            weight_norm = tf.math.maximum(
                tf.norm(weight, axis=axis, keepdims=True), eps
            )
            grad_norm = tf.norm(grad, axis=axis, keepdims=True)
        else:
            # no gradient clipping for biases and other 1-D or scalar parameters
            return grad
        max_norm = weight_norm * lamb
        mask = grad_norm > max_norm
        clipped_grad = max_norm * (
            grad / tf.math.maximum(grad_norm, 1e-6)
        )  # just in case
        # if |G| > |W| * lamb: G <- lamb * |W| * G/|G|     where |x| = norm along row of x
        # else: G <- G
        return tf.where(mask, clipped_grad, grad)

    @tf.function
    def update_weights(self, grads):
        # split up weights for each optimizer
        gradients = {"AdamW": [], "Adam": []}
        weights = {"AdamW": [], "Adam": []}

        for grad, W in zip(grads, self.model.trainable_weights):
            if grad is None:
                raise ValueError(
                    f"No gradient found for {W.name} most likely because the weight is not used in the model."
                )

            # divide gradients of shared layers
            if "shared" in W.name:
                nBlocks = self.model.num_blocks
                # one more output block than interaction block
                if "out" in W.name:
                    nBlocks = nBlocks + 1
                if isinstance(grad, tf.IndexedSlices):
                    grad = tf.IndexedSlices(
                        grad.values / nBlocks,
                        grad.indices,
                        dense_shape=grad.dense_shape,
                    )
                else:
                    grad = grad / nBlocks

            # adaptive gradient clipping
            if self.agc:
                grad = self._adaptive_gradient_clipping(self.grad_clip_max, grad, W)

            # split up weights for each optimizer
            if "kernel" in W.name:
                gradients["AdamW"] += [grad]
                weights["AdamW"] += [W]
            else:
                gradients["Adam"] += [grad]
                weights["Adam"] += [W]

        if not self.agc:
            global_norm = tf.linalg.global_norm(gradients["AdamW"])
            for var in ["Adam", "AdamW"]:
                gradients[var], _ = tf.clip_by_global_norm(
                    gradients[var], self.grad_clip_max, use_norm=global_norm
                )

        for var in ["Adam", "AdamW"]:
            self.optimizers["Avg" + var].apply_gradients(
                zip(gradients[var], weights[var])
            )  # updates lr as well

    def load_averaged_variables(self):
        # group weights depending on optimizer if not already done so
        self.weight_groups = self._group_weights_maybe(self.weight_groups)
        # assign running average
        self.optimizers["AvgAdamW"].assign_average_vars(self.weight_groups["AdamW"])
        self.optimizers["AvgAdam"].assign_average_vars(self.weight_groups["Adam"])

    def save_variable_backups(self):
        if self.backup_vars is None:
            self.backup_vars = self._get_variable_copy(self.model.trainable_weights)
        else:
            for var, bck in zip(self.model.trainable_weights, self.backup_vars):
                bck.assign(var)

    def restore_variable_backups(self):
        for var, bck in zip(self.model.trainable_weights, self.backup_vars):
            var.assign(bck)

    def get_mae(self, targets, mean_pred):
        """
        Mean Absolute Error
        """
        return tf.reduce_mean(tf.abs(targets - mean_pred))

    def get_l2mae(self, targets, mean_pred):
        """
        Mean Error of L2 norm
        """
        return tf.reduce_mean(tf.norm(targets - mean_pred, ord=2, axis=1))

    def get_nll(self, targets, mean_pred, var_pred):
        nll = 0.5 * (
            tf.math.log(var_pred) + tf.math.square(targets - mean_pred) / var_pred
        )
        nll = tf.reduce_mean(nll)
        return nll

    def predict(self, inputs, training=False):
        epsilon = 1e-7

        energy, forces = self.model(inputs, training=training)

        if self.mve:
            mean_energy = energy[:, :1]
            var_energy = tf.nn.softplus(energy[:, 1:]) + epsilon
            mean_forces = forces[:, 0, :]
            var_forces = tf.nn.softplus(forces[:, 1, :]) + epsilon
            return mean_energy, var_energy, mean_forces, var_forces
        else:
            if len(tf.shape(forces)) == 3:
                forces = forces[:, 0, :]
            return energy, None, forces, None

    @tf.function
    def predict_on_batch(self, dataset_iter):
        inputs, _ = next(dataset_iter)
        return self.predict(inputs, training=False)

    @tf.function
    def train_on_batch(self, dataset_iter, metrics):
        inputs, targets = next(dataset_iter)

        with tf.GradientTape() as tape:
            mean_energy, var_energy, mean_forces, var_forces = self.predict(
                inputs, training=True
            )

            if self.mve:
                energy_nll = self.get_nll(targets["E"], mean_energy, var_energy)
                force_nll = self.get_nll(targets["F"], mean_forces, var_forces)
                loss = energy_nll * (1 - self.rho_force) + self.rho_force * force_nll
            else:
                energy_mae = self.get_mae(targets["E"], mean_energy)
                if self.loss == "mae":
                    force_metric = self.get_mae(targets["F"], mean_forces)
                else:
                    force_metric = self.get_l2mae(targets["F"], mean_forces)
                loss = energy_mae * (1 - self.rho_force) + self.rho_force * force_metric

        gradients = tape.gradient(loss, self.model.trainable_weights)
        self.update_weights(gradients)

        if self.mve:
            energy_mae = self.get_mae(targets["E"], mean_energy)
            force_mae = self.get_mae(targets["F"], mean_forces)
            force_rmse = self.get_l2mae(targets["F"], mean_forces)

        else:
            if self.loss == "mae":
                force_mae = force_metric
                force_rmse = self.get_l2mae(targets["F"], mean_forces)
            else:
                force_mae = self.get_mae(targets["F"], mean_forces)
                force_rmse = force_metric

        if self.mve:
            # update molecule metrics
            metrics.update_state(
                nsamples=tf.shape(mean_energy)[0],
                loss=loss,
                energy_mae=energy_mae,
                energy_nll=energy_nll,
                energy_var=var_energy,
            )
            # update atom metrics
            metrics.update_state(
                nsamples=tf.shape(mean_forces)[0],
                force_mae=force_mae,  # global mae
                force_rmse=force_rmse,
                force_nll=force_nll,
                force_var=var_forces,
            )
        else:
            # update molecule metrics
            metrics.update_state(
                nsamples=tf.shape(mean_energy)[0],
                loss=loss,
                energy_mae=energy_mae,
            )
            # update atom metrics
            metrics.update_state(
                nsamples=tf.shape(mean_forces)[0],
                force_mae=force_mae,
                force_rmse=force_rmse,
            )

        return loss

    @tf.function
    def test_on_batch(self, dataset_iter, metrics):
        inputs, targets = next(dataset_iter)
        mean_energy, var_energy, mean_forces, var_forces = self.predict(
            inputs, training=False
        )

        energy_mae = self.get_mae(targets["E"], mean_energy)
        force_mae = self.get_mae(targets["F"], mean_forces)
        force_rmse = self.get_l2mae(targets["F"], mean_forces)

        if self.mve:
            energy_nll = self.get_nll(targets["E"], mean_energy, var_energy)
            loss = energy_nll * (1 - self.rho_force) + self.rho_force * force_mae
            force_nll = self.get_nll(targets["F"], mean_forces, var_forces)
            loss = energy_nll * (1 - self.rho_force) + self.rho_force * force_nll

            # update molecule metrics
            metrics.update_state(
                nsamples=tf.shape(mean_energy)[0],
                loss=loss,
                energy_mae=energy_mae,
                energy_nll=energy_nll,
                energy_var=var_energy,
            )
            # update atom metrics
            metrics.update_state(
                nsamples=tf.shape(mean_forces)[0],
                force_mae=force_mae,
                force_rmse=force_rmse,
                force_nll=force_nll,
                force_var=var_forces,
            )

        else:
            force_metric = force_mae if self.loss == "mae" else force_rmse
            loss = (1 - self.rho_force) * energy_mae + self.rho_force * force_metric

            # update molecule metrics
            metrics.update_state(
                nsamples=tf.shape(mean_energy)[0],
                loss=loss,
                energy_mae=energy_mae,
            )
            # update atom metrics
            metrics.update_state(
                nsamples=tf.shape(mean_forces)[0],
                force_mae=force_mae,
                force_rmse=force_rmse,
            )

        return loss

    @tf.function
    def eval_on_batch(self, dataset_iter):
        inputs, targets = next(dataset_iter)
        energy, _, forces, _ = self.predict(inputs, training=False)
        return (energy, forces), targets


class ReduceLROnPlateau:
    """
    Reduce learning rate on plateau if loss did not decrease (significantly) for some time.

    Parameters
    ----------
        trainer: Trainer
            Traininer instacne.
        patience: int
            Number of evaluation intervals after decaying the learning rate.
        decay_factor: float
            Multiply inverse of decay factor by learning rate to obtain new learning rate.
        min_delta: float
            Minimum decreace in loss else treat as no progress.
        max_reduce: int
            How often to reduce on plateau at most.
        cooldown: int
            Number of evaluation intervals after which to return to normal operation.
    """

    def __init__(
        self,
        trainer,
        patience,  # in steps
        decay_factor,  # decay_factor
        min_delta=1e-4,  # minimum difference else assert no progress
        max_reduce=10,
        cooldown=0,
    ):
        self.decay_factor = tf.constant(decay_factor, dtype=tf.float32)
        self.patience = patience
        self.min_delta = min_delta
        self.trainer = trainer
        self.max_reduce = max_reduce

        self.best = np.inf
        self.wait = patience
        self.cooldown = cooldown

        self._reduce_counter = 0
        self._cooldown_counter = 0

    def decay_maybe(self, val_loss):
        # still in cooldown
        if self._cooldown_counter > 0:
            self._cooldown_counter -= 1
            return

        if val_loss >= self.best - self.min_delta:
            # reduce if only improved very little (if at all)
            self.wait -= 1
        else:
            # reset
            self.wait = self.patience
            self.best = val_loss

        if (
            (self.wait == 0)
            and (self.decay_factor != 1)
            and (self._reduce_counter < self.max_reduce)
        ):
            # reset
            self.wait = self.patience
            self._reduce_counter += 1
            self._cooldown_counter = self.cooldown

            # decay learning rate by factor
            for opt in ["AvgAdam", "AvgAdamW"]:
                optimizer = self.trainer.optimizers[opt]
                if hasattr(
                    optimizer.learning_rate, "initial_learning_rate"
                ):  # callable learning rate
                    optimizer.learning_rate.initial_learning_rate = (
                        optimizer.learning_rate.initial_learning_rate
                        * self.decay_factor
                    )
                else:
                    optimizer.learning_rate = (
                        optimizer.learning_rate * self.decay_factor
                    )
            logging.info(f"Reduced lr on plateau for the {self._reduce_counter}. time.")
