import numpy as np
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
os.environ["AUTOGRAPH_VERBOSITY"] = "1"
import logging
import string
import random
import time
from datetime import datetime

from gemnet.model.gemnet import GemNet
from gemnet.training.trainer import Trainer
from gemnet.training.metrics import Metrics, BestMetrics
from gemnet.training.data_container import DataContainer
from gemnet.training.data_provider import DataProvider

from sacred import Experiment
from tensorflow.python.keras.utils.layer_utils import count_params

import seml
import tensorflow as tf

ex = Experiment()
seml.setup_logger(ex)

# TensorFlow logging verbosity
tf.get_logger().setLevel("WARN")
tf.autograph.set_verbosity(1)


@ex.post_run_hook
def collect_stats(_run):
    seml.collect_exp_stats(_run)


@ex.config
def config():
    overwrite = None
    db_collection = None
    if db_collection is not None:
        ex.observers.append(
            seml.create_mongodb_observer(db_collection, overwrite=overwrite)
        )


@ex.automain
def run(
    num_spherical,
    num_radial,
    num_blocks,
    emb_size_atom,
    emb_size_edge,
    emb_size_trip,
    emb_size_quad,
    emb_size_rbf,
    emb_size_cbf,
    emb_size_sbf,
    num_before_skip,
    num_after_skip,
    num_concat,
    num_atom,
    emb_size_bil_quad,
    emb_size_bil_trip,
    triplets_only,
    forces_coupled,
    direct_forces,
    mve,
    cutoff,
    int_cutoff,
    envelope_exponent,
    extensive,
    output_init,
    scale_file,
    data_seed,
    dataset,
    val_dataset,
    num_train,
    num_val,
    logdir,
    loss,
    tfseed,
    num_steps,
    rho_force,
    ema_decay,
    weight_decay,
    grad_clip_max,
    agc,
    decay_patience,
    decay_factor,
    decay_cooldown,
    batch_size,
    evaluation_interval,
    patience,
    save_interval,
    learning_rate,
    warmup_steps,
    decay_steps,
    decay_rate,
    staircase,
    restart,
    comment,
    ):

    physical_devices = tf.config.list_physical_devices("GPU")
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    tf.random.set_seed(tfseed)

    logging.info("Start training")
    # log hyperparameters
    logging.info(
        "Hyperparams: \n" + "\n".join(f"{key}: {val}" for key, val in locals().items())
    )
    gpus = list(physical_devices)
    logging.info(f"Available GPUs: {gpus}")
    if len(gpus) == 0:
        logging.warning("No GPUs were found. Training is run on CPU!")

    # Used for creating a "unique" id for a run (almost impossible to generate the same twice)
    def id_generator(
        size=6, chars=string.ascii_uppercase + string.ascii_lowercase + string.digits
    ):
        return "".join(random.SystemRandom().choice(chars) for _ in range(size))

    # A unique directory name is created for this run based on the input
    if (restart is None) or (restart == "None"):
        directory = (
            logdir
            + "/"
            + datetime.now().strftime("%Y%m%d_%H%M%S")
            + "_"
            + id_generator()
            + "_"
            + os.path.basename(dataset)
            + "_"
            + str(comment)
        )
    else:
        directory = restart

    logging.info(f"Directory: {directory}")
    logging.info("Create directories")

    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    best_dir = os.path.join(directory, "best")
    if not os.path.exists(best_dir):
        os.makedirs(best_dir)
    log_dir = os.path.join(directory, "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    best_ckpt_file = os.path.join(best_dir, "ckpt")
    step_ckpt_folder = log_dir

    logging.info("Initialize model")
    model = GemNet(
        num_spherical=num_spherical,
        num_radial=num_radial,
        num_blocks=num_blocks,
        emb_size_atom=emb_size_atom,
        emb_size_edge=emb_size_edge,
        emb_size_trip=emb_size_trip,
        emb_size_quad=emb_size_quad,
        emb_size_rbf=emb_size_rbf,
        emb_size_cbf=emb_size_cbf,
        emb_size_sbf=emb_size_sbf,
        num_before_skip=num_before_skip,
        num_after_skip=num_after_skip,
        num_concat=num_concat,
        num_atom=num_atom,
        emb_size_bil_quad=emb_size_bil_quad,
        emb_size_bil_trip=emb_size_bil_trip,
        num_targets=2 if mve else 1,
        triplets_only=triplets_only,
        direct_forces=direct_forces,
        forces_coupled=forces_coupled,
        cutoff=cutoff,
        int_cutoff=int_cutoff,
        envelope_exponent=envelope_exponent,
        activation="swish",
        extensive=extensive,
        output_init=output_init,
        scale_file=scale_file,
    )

    # Initialize summary writer
    summary_writer = tf.summary.create_file_writer(log_dir)
    train = {}
    validation = {}

    logging.info("Load dataset")
    data_container = DataContainer(
        dataset, cutoff=cutoff, int_cutoff=int_cutoff, triplets_only=triplets_only
    )

    if val_dataset is not None:
        # Initialize DataProvider
        if num_train == 0:
            num_train = len(data_container)
        logging.info(f"Training data size: {num_train}")
        train["data_provider"] = DataProvider(
            data_container,
            num_train,
            0,
            batch_size,
            seed=data_seed,
            shuffle=True,
            random_split=True,
        )

        # Initialize validation dataset
        val_data_container = DataContainer(
            val_dataset,
            cutoff=cutoff,
            int_cutoff=int_cutoff,
            triplets_only=triplets_only,
        )
        if num_val == 0:
            num_val = len(val_data_container)
        logging.info(f"Validation data size: {num_val}")
        validation["data_provider"] = DataProvider(
            val_data_container,
            0,
            num_val,
            batch_size,
            seed=data_seed,
            shuffle=True,
            random_split=True,
        )
    else:
        # Initialize DataProvider (splits dataset into 3 sets based on data_seed)
        logging.info(f"Training data size: {num_train}")
        logging.info(f"Validation data size: {num_val}")
        assert num_train > 0
        assert num_val > 0
        train["data_provider"] = DataProvider(
            data_container,
            num_train,
            num_val,
            batch_size,
            seed=data_seed,
            shuffle=True,
            random_split=True,
        )
        validation["data_provider"] = train["data_provider"]

    train["dataset_iter"] = train["data_provider"].get_dataset("train")
    validation["dataset_iter"] = validation["data_provider"].get_dataset("val")

    with summary_writer.as_default():
        logging.info("Prepare training")
        # Initialize trainer
        trainer = Trainer(
            model,
            learning_rate=learning_rate,
            decay_steps=decay_steps,
            decay_rate=decay_rate,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            ema_decay=ema_decay,
            decay_patience=decay_patience,
            decay_factor=decay_factor,
            decay_cooldown=decay_cooldown,
            grad_clip_max=grad_clip_max,
            rho_force=rho_force,
            mve=mve,
            loss=loss,
            staircase=staircase,
            agc=agc,
        )

        # Initialize metrics
        train["metrics"] = Metrics("train", trainer.tracked_metrics, ex)
        validation["metrics"] = Metrics("val", trainer.tracked_metrics, ex)

        # Save/load best recorded loss (only the best model is saved)
        metrics_best = BestMetrics(best_dir, validation["metrics"])

        # Set up checkpointing
        ckpt = tf.train.Checkpoint(
            step=tf.Variable(1), model=model, **trainer.optimizers
        )
        manager = tf.train.CheckpointManager(ckpt, step_ckpt_folder, max_to_keep=1)

        # Restore latest checkpoint
        ckpt_restored = tf.train.latest_checkpoint(log_dir)
        if ckpt_restored is not None:
            ckpt.restore(ckpt_restored)
            # reset the optimizer to not restore the saved learning rate
            # trainer.reset_optimizer(warmup_steps, learning_rate, decay_steps, decay_rate)
            metrics_best.restore()
            logging.info(f"Restored best metrics: {metrics_best.loss}")
            step_init = ckpt.step.numpy()
        else:
            metrics_best.inititalize()
            step_init = 0

        logging.info(f"Checkpoint restoration status: {ckpt_restored}")

        if ex is not None:
            ex.current_run.info = {"directory": directory}

        # Training loop
        logging.info("Start training")

        params_counted = False
        steps_per_epoch = int(np.ceil(num_train / batch_size))

        for step in range(step_init + 1, num_steps + 1):

            # Update step number
            ckpt.step.assign(step)
            tf.summary.experimental.set_step(step)
            if ex is not None:
                # start after evaluation to not include time on validation set
                if step == evaluation_interval + 1:
                    start = time.perf_counter()
                if step == 2 * evaluation_interval - 1:
                    end = time.perf_counter()
                    time_delta = end - start
                    nsteps = evaluation_interval - 2
                    ex.current_run.info.update(
                        {"seconds_per_step": time_delta / nsteps,
                        "min_per_epoch": int(time_delta / nsteps * steps_per_epoch * 100 / 60) / 100  # two digits only
                        }
                    )

            # keep track of the learning rate
            if step % 10 == 0:
                tf.summary.scalar(
                    "lr", trainer.optimizers["AdamW"]._decayed_lr(tf.float32)
                )

            # Perform training step
            trainer.train_on_batch(train["dataset_iter"], train["metrics"])

            # Save progress
            if step % save_interval == 0:
                manager.save()

            # Check performance on the validation set
            if step % evaluation_interval == 0:

                # Save backup variables and load averaged variables
                trainer.save_variable_backups()
                trainer.load_averaged_variables()

                # Compute averages
                for i in range(int(np.ceil(num_val / batch_size))):
                    trainer.test_on_batch(
                        validation["dataset_iter"], validation["metrics"]
                    )

                # Update and save best result
                if validation["metrics"].loss < metrics_best.loss:
                    metrics_best.update(step, validation["metrics"])
                    model.save_weights(best_ckpt_file)

                # save the number of parameters
                if (not params_counted) and (ex is not None):
                    # params can only be counted if weights are already built -> therefore it is here
                    params_counted = True
                    nparams = count_params(model.trainable_weights)
                    ex.current_run.info.update({"nParams": nparams})

                # write to summary writer
                metrics_best.write()

                epoch = step // steps_per_epoch
                train_metrics_res = train["metrics"].result(append_tag=False)
                val_metrics_res = validation["metrics"].result(append_tag=False)
                metrics_strings = [
                    f"{key}: train={train_metrics_res[key]:.6f}, val={val_metrics_res[key]:.6f}"
                    for key in validation["metrics"].keys
                ]
                logging.info(
                    f"{step}/{num_steps} (epoch {epoch}): " + "; ".join(metrics_strings)
                )

                # decay learning rate on plateau
                trainer.decay_maybe(validation["metrics"].loss)

                train["metrics"].write()
                validation["metrics"].write()
                train["metrics"].reset_states()
                validation["metrics"].reset_states()

                # Restore backup variables
                trainer.restore_variable_backups()

                # early stopping
                if (patience > 0) and (step - metrics_best.step > patience * evaluation_interval):
                    break

    return {key + "_best": val for key, val in metrics_best.items()}
