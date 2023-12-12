# 2nd script

# Dataset folder
# trainhopper5frames
# scripts
#!/usr/bin/env python3
import os
import sys

import numpy as np
from dotenv import load_dotenv

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import time
from collections import namedtuple
from glob import glob

# from datasets.create_physics_scenes import PARTICLE_RADIUS
import horovod.tensorflow as hvd
import tensorflow as tf
from evaluate_network import evaluate_tf as evaluate

from data_prep.dataset_reader_physics import read_data_train, read_data_val
from utils.deeplearningutilities.tf import MyCheckpointManager, Trainer

# from tensorflow.python.client import device_lib

# def get_available_devices():
#     local_device_protos = device_lib.list_local_devices()
#     return [x.name for x in local_device_protos]

# print(get_available_devices())
# variables
GPU_PER_NODE = 2
# Initialize Horovod
hvd.init()

# Pin GPU to be used to process local rank (one GPU per process)
# gpus = tf.config.experimental.list_physical_devices("GPU")
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)
# if gpus:
#     tf.config.experimental.set_visible_devices(
#         gpus[hvd.local_rank() % GPU_PER_NODE], "GPU"
#     )

# print("pid={:d}, total processes={:d}".format(hvd.rank(), hvd.size()))
# the train dir stores all checkpoints and summaries. The dir name is the name of this file without .py
load_dotenv()
output_scenes_dir = os.getenv("OUTPUT_SCENES_DIR")
dataset_path = f"{output_scenes_dir}_data"
# dataset_path = "/Users/bekzatajan/PhD/Thesis/mldem_rebuilt/data/exp_1_data"
train_dir = os.path.join(dataset_path, "huinia")
lluTimeStep = 0.02
lluFilter = 6
PARTICLE_RADIUS = 0.025

# dataset_path = os.path.join(os.path.dirname(__file__), '..', 'datasets',
#                            'ours_default_data')

val_files = sorted(glob(os.path.join(dataset_path, "valid", "*.zst")))
train_files = sorted(glob(os.path.join(dataset_path, "train", "*.zst")))

_k = 200

TrainParams = namedtuple("TrainParams", ["max_iter", "base_lr", "batch_size"])
train_params = TrainParams(50 * _k // hvd.size(), 0.001 * hvd.size(), 16)


def create_model():
    from models.default_tf import MyParticleNetwork

    """Returns an instance of the network for training and evaluation"""
    model = MyParticleNetwork(
        particle_radius=PARTICLE_RADIUS, filter_scale=lluFilter, timestep=lluTimeStep
    )

    return model


def main():
    val_dataset = read_data_val(files=val_files, window=1, cache_data=True)

    dataset = read_data_train(
        files=train_files,
        batch_size=train_params.batch_size,
        window=6,
        random_rotation=True,
        num_workers=2,
    )
    data_iter = iter(dataset)
    breakpoint()

    trainer = Trainer(train_dir)

    model = create_model()

    boundaries = [
        25 * _k,
        30 * _k,
        35 * _k,
        40 * _k,
        45 * _k,
    ]
    lr_values = [
        train_params.base_lr * 1.0,
        train_params.base_lr * 0.5,
        train_params.base_lr * 0.25,
        train_params.base_lr * 0.125,
        train_params.base_lr * 0.5 * 0.125,
        train_params.base_lr * 0.25 * 0.125,
    ]
    learning_rate_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries, lr_values
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_fn, epsilon=1e-6)

    checkpoint = tf.train.Checkpoint(
        step=tf.Variable(0), model=model, optimizer=optimizer
    )

    manager = MyCheckpointManager(
        checkpoint,
        trainer.checkpoint_dir,
        keep_checkpoint_steps=list(range(1 * _k, train_params.max_iter + 1, 1 * _k)),
    )

    def euclidean_distance(a, b, epsilon=1e-9):
        return tf.sqrt(tf.reduce_sum((a - b) ** 2, axis=-1) + epsilon)

    def loss_fn(pr_pos, gt_pos, num_fluid_neighbors):
        micro = tf.reduce_mean(euclidean_distance(pr_pos, gt_pos))
        a_pr_pos = tf.reduce_mean(pr_pos, axis=0)
        a_gt_pos = tf.reduce_mean(gt_pos, axis=0)
        # tf.print(a_gt_pos)
        macro = euclidean_distance(a_pr_pos, a_gt_pos)
        return micro * 0.5 + macro * 0.5

    @tf.function(experimental_relax_shapes=True)
    def train(model, batch, first_batch):
        with tf.GradientTape() as tape:
            losses = []

            batch_size = train_params.batch_size
            for batch_i in range(batch_size):
                inputs = [
                    batch["pos0"][batch_i],
                    batch["vel0"][batch_i],
                    None,
                    batch["box"][batch_i],
                    batch["box_normals"][batch_i],
                ]

                pr_pos1, pr_vel1 = model(inputs)

                l = 0.1 * loss_fn(
                    pr_pos1, batch["pos1"][batch_i], model.num_fluid_neighbors
                )

                inputs = (
                    pr_pos1,
                    pr_vel1,
                    None,
                    batch["box"][batch_i],
                    batch["box_normals"][batch_i],
                )
                pr_pos2, pr_vel2 = model(inputs)

                l += 0.1 * loss_fn(
                    pr_pos2, batch["pos2"][batch_i], model.num_fluid_neighbors
                )

                inputs = (
                    pr_pos2,
                    pr_vel2,
                    None,
                    batch["box"][batch_i],
                    batch["box_normals"][batch_i],
                )
                pr_pos3, pr_vel3 = model(inputs)

                l += 0.1 * loss_fn(
                    pr_pos3, batch["pos3"][batch_i], model.num_fluid_neighbors
                )
                # next frame
                inputs = (
                    pr_pos3,
                    pr_vel3,
                    None,
                    batch["box"][batch_i],
                    batch["box_normals"][batch_i],
                )
                pr_pos4, pr_vel4 = model(inputs)

                l += 0.1 * loss_fn(
                    pr_pos4, batch["pos4"][batch_i], model.num_fluid_neighbors
                )
                # next frame
                inputs = (
                    pr_pos4,
                    pr_vel4,
                    None,
                    batch["box"][batch_i],
                    batch["box_normals"][batch_i],
                )
                pr_pos5, pr_vel5 = model(inputs)

                l += 0.6 * loss_fn(
                    pr_pos5, batch["pos5"][batch_i], model.num_fluid_neighbors
                )
                losses.append(l)

            losses.extend(model.losses)
            total_loss = 128 * tf.add_n(losses) / batch_size

        # Horovod: add Horovod Distributed GradientTape.
        tape = hvd.DistributedGradientTape(tape)

        grads = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # Horovod: broadcast initial variable states from rank 0 to all other processes.
        # This is necessary to ensure consistent initialization of all workers when
        # training is started with random weights or restored from a checkpoint.
        #
        # Note: broadcast should be done after the first gradient step to ensure optimizer
        # initialization.
        if first_batch:
            hvd.broadcast_variables(model.variables, root_rank=0)
            hvd.broadcast_variables(optimizer.variables(), root_rank=0)

        return total_loss

    if manager.latest_checkpoint:
        print("restoring from ", manager.latest_checkpoint)
        checkpoint.restore(manager.latest_checkpoint)

    display_str_list = []
    first_batch = True
    while trainer.keep_training(
        checkpoint.step,
        train_params.max_iter,
        checkpoint_manager=manager,
        display_str_list=display_str_list,
        pid=hvd.rank(),
    ):
        data_fetch_start = time.time()
        batch = next(data_iter)

        # print("step={:} checkpoint={:}".format(trainer.current_step,manager.latest_checkpoint))
        batch_tf = {}
        for k in (
            "pos0",
            "vel0",
            "pos1",
            "pos2",
            "pos3",
            "pos4",
            "pos5",
            "box",
            "box_normals",
        ):
            batch_tf[k] = [tf.convert_to_tensor(x) for x in batch[k]]
        data_fetch_latency = time.time() - data_fetch_start
        trainer.log_scalar_every_n_minutes(5, "DataLatency", data_fetch_latency)

        current_loss = train(model, batch_tf, first_batch)
        display_str_list = ["loss", float(current_loss)]
        first_batch = False

        if trainer.current_step % 10 == 0:
            with trainer.summary_writer.as_default():
                tf.summary.scalar("TotalLoss", current_loss)
                tf.summary.scalar("LearningRate", optimizer.lr(trainer.current_step))
        if trainer.current_step % 1000 == 0:
            if hvd.local_rank() == 0:
                model.save_weights(
                    train_dir + "/model_weights_{:d}.h5".format(trainer.current_step)
                )

        if trainer.current_step % (1 * _k) == 0:
            for k, v in evaluate(model, val_dataset, frame_skip=20).items():
                with trainer.summary_writer.as_default():
                    tf.summary.scalar("eval/" + k, v)
    if hvd.local_rank() == 0:
        model.save_weights(train_dir + "/model_weights.h5")

    if trainer.current_step == train_params.max_iter:
        return trainer.STATUS_TRAINING_FINISHED
    else:
        return trainer.STATUS_TRAINING_UNFINISHED


if __name__ == "__main__":
    import multiprocessing as mp

    mp.set_start_method("spawn")
    sys.exit(main())
