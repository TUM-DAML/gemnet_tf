import tensorflow as tf


class LinearWarmupExponentialDecay(tf.optimizers.schedules.LearningRateSchedule):
    """This schedule combines a linear warmup with an exponential decay.

    Parameters
    ----------
        learning_rate: float
            Initial learning rate.
        decay_steps: float
            Number of steps until learning rate reaches learning_rate*decay_rate
        decay_rate: float
            Decay rate.
        warmup_steps: int
            Total number of warmup steps of the learning rate schedule.
        staircase: bool
            If True use staircase decay and not (continous) exponential decay.
    """

    def __init__(
        self,
        learning_rate: float,
        warmup_steps: int,
        decay_steps: int,
        decay_rate: float,
        staircase: bool,
    ):
        super().__init__()
        if warmup_steps == 0:
            warmup_steps = 1
        self.warmup = tf.optimizers.schedules.PolynomialDecay(
            1 / warmup_steps, warmup_steps, end_learning_rate=1
        )
        self.decay = tf.optimizers.schedules.ExponentialDecay(
            learning_rate, decay_steps, decay_rate, staircase=staircase
        )

    def __call__(self, step):
        return self.warmup(step) * self.decay(step)

    @property
    def initial_learning_rate(self):
        return self.decay.initial_learning_rate

    @initial_learning_rate.setter
    def initial_learning_rate(self, value):
        self.decay.initial_learning_rate = value
