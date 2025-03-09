import tensorflow as tf
from tensorflow import keras
import argparse
from m4depth_options import M4DepthOptions
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import dataloaders as dl
from callbacks import *
from cosemdepth_network import *
from metrics import *

if __name__ == '__main__':
    cmdline = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    model_opts = M4DepthOptions(cmdline)
    cmd, test_args = cmdline.parse_known_args()

    # configure tensorflow gpus
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        pass

    enable_validation = cmd.enable_validation
    try:
        # Manage GPU memory to be able to run the validation step in parallel on the same GPU
        if cmd.mode == "validation":
            print('limit memory')
            tf.config.set_logical_device_configuration(physical_devices[0],
                                                       [tf.config.LogicalDeviceConfiguration(memory_limit=1200)])
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        print("GPUs initialization failed")
        enable_validation = False
        pass

    working_dir = os.getcwd()

    topair_params = dl.DataloaderParameters(model_opts.dataloader_settings.db_path_config,
                                           os.path.join(*[cmd.records_path, "topair", "train_data"]),
                                           4, 4, True, os.path.join(*[cmd.records_path, "topair", "val_data"]))
    topair_dataloader = dl.get_loader("topair")
    topair_dataloader.get_dataset("train", topair_params, batch_size=cmd.batch_size)
    

    midair_params = dl.DataloaderParameters(model_opts.dataloader_settings.db_path_config,
                                           os.path.join(*[cmd.records_path, "midair", "train_data"]),
                                           4, 4, True, os.path.join(*[cmd.records_path, "midair", "val_data"]))
    midair_dataloader = dl.get_loader("midair")
    midair_dataloader.get_dataset("train", midair_params, batch_size=cmd.batch_size)
    joint_dataset_length = topair_dataloader.length*2
    joint_dataset = tf.data.Dataset.sample_from_datasets([topair_dataloader.dataset.repeat(), midair_dataloader.dataset.repeat()], weights=[0.5, 0.5], stop_on_empty_dataset=True)
    
    print("LENGTH ", joint_dataset_length)
    #joint_dataset = tf.data.Dataset.sample_from_datasets([midair_dataloader.dataset.repeat()], weights=[1.0], stop_on_empty_dataset=True)
    
    seq_len = cmd.seq_len
    nbre_levels = cmd.arch_depth
    ckpt_dir = cmd.ckpt_dir

    # print("Training on %s" % cmd.dataset)
    tf.random.set_seed(42)
    data = joint_dataset
    
    model = JointM4DepthSemantic(nbre_levels=nbre_levels,
                    ablation_settings=model_opts.ablation_settings,
                    is_training=True, num_classes = midair_dataloader.class_count)

    tensorboard_cbk = keras.callbacks.TensorBoard(
        log_dir=cmd.log_dir, histogram_freq=1200, write_graph=True,
        write_images=False, update_freq=1200,
        profile_batch=0, embeddings_freq=0, embeddings_metadata=None)
    # stop_nan_cbk = ks.callbacks.TerminateOnNaN()
    model_checkpoint_cbk = CustomCheckpointCallback(os.path.join(ckpt_dir,"train"), resume_training=True)

    opt = tf.keras.optimizers.Adam(learning_rate=0.0001)

    #model.compile(optimizer=opt, metrics=[RootMeanSquaredLogError()])
    model.compile(optimizer=opt)

    if enable_validation:
        val_cbk = [CustomJointMidairTopairValidationCallback(cmd, args=test_args)]
    else:
        val_cbk = []
    
    #model.fit(data, epochs=model_checkpoint_cbk.resume_epoch + (20000 // joint_dataset_length) + 1,
    model.fit(data, epochs=model_checkpoint_cbk.resume_epoch + 20,
                initial_epoch=model_checkpoint_cbk.resume_epoch,
                steps_per_epoch= joint_dataset_length,
                callbacks=[tensorboard_cbk, model_checkpoint_cbk] + val_cbk)
