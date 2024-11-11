"""
----------------------------------------------------------------------------------------
Copyright (c) 2022 - Michael Fonder, University of Liège (ULiège), Belgium.

This program is free software: you can redistribute it and/or modify it under the terms
of the GNU Affero General Public License as published by the Free Software Foundation,
either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU Affero General Public License for more details.
You should have received a copy of the GNU Affero General Public License along with this
program. If not, see < [ https://www.gnu.org/licenses/ | https://www.gnu.org/licenses/ ] >.
----------------------------------------------------------------------------------------
"""

import os
import argparse
from m4depth_options import M4DepthOptions

cmdline = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
model_opts = M4DepthOptions(cmdline)
cmd, test_args = cmdline.parse_known_args()
if cmd.mode == 'eval':
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit'
    print("PROBLEM")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import tensorflow as tf
from tensorflow import keras
import numpy as np
import dataloaders as dl
from callbacks import *
from joint_m4depth_network import *
#from metrics import *
#from joint_m4depth_network_fullres import *
import time
from PIL import Image

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
        print("error in gpu")
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
    print("The current working directory is : %s" % working_dir)
    #tf.config.run_functions_eagerly(True)
    
    chosen_dataloader = dl.get_loader(cmd.dataset)
    #val_dataloader = dl.get_loader(cmd.dataset)
    seq_len = cmd.seq_len
    nbre_levels = cmd.arch_depth
    ckpt_dir = cmd.ckpt_dir

    if cmd.mode == 'train' or cmd.mode == 'finetune':

        print("Training on %s" % cmd.dataset)
        tf.random.set_seed(42)
        chosen_dataloader.get_dataset("train", model_opts.dataloader_settings, batch_size=cmd.batch_size)
        data = chosen_dataloader.dataset
        
        
        #val_dataloader.get_dataset("valid", model_opts.dataloader_settings, batch_size=cmd.batch_size)
        #val_data = val_dataloader.dataset
        
        model = JointM4DepthSemantic(nbre_levels=nbre_levels,
                        ablation_settings=model_opts.ablation_settings,
                        is_training=True, num_classes = chosen_dataloader.class_count)
        print("TRAIN Length of the dataset = ", chosen_dataloader.length)
        #print("VALID Length of the dataset = ", val_dataloader.length)
        # Initialize callbacks
        tensorboard_cbk = keras.callbacks.TensorBoard(
            log_dir=cmd.log_dir, histogram_freq=1200, write_graph=True,
            write_images=False, update_freq=1200,
            profile_batch=0, embeddings_freq=0, embeddings_metadata=None)
        model_checkpoint_cbk = CustomCheckpointCallback(os.path.join(ckpt_dir,"train"), resume_training=True)
        
        #lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.0001, decay_steps=50000, decay_rate=0.5)
        #opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        opt = tf.keras.optimizers.Adam(learning_rate=0.0001)

        
        #model.compile(optimizer=opt, metrics=[tf.keras.metrics.MeanIoU(num_classes = chosen_dataloader.class_count)], run_eagerly = True)
        #model.compile(optimizer=opt, metrics=[RootMeanSquaredLogError(), tf.keras.metrics.MeanIoU(num_classes = chosen_dataloader.class_count)])
        model.compile(optimizer=opt)
        
        print("Length of the dataset = ", chosen_dataloader.length)
        if enable_validation:
            val_cbk = [CustomMidairValidationCallback(cmd, args=test_args)]
        else:
            val_cbk = []

        # Adapt number of steps depending on desired usecase
        if cmd.mode == 'finetune':
            nbre_epochs = model_checkpoint_cbk.resume_epoch + (40000 // chosen_dataloader.length)
        else:
            #nbre_epochs = (350000 // chosen_dataloader.length)
            nbre_epochs = (0 // chosen_dataloader.length)

        #model.fit(data, epochs= nbre_epochs + 1,
        #          initial_epoch=model_checkpoint_cbk.resume_epoch,
        #          callbacks=[tensorboard_cbk, model_checkpoint_cbk] + val_cbk, validation_data = val_data, validation_freq= 2)
        model.fit(data, epochs= nbre_epochs + 1,
                  initial_epoch=model_checkpoint_cbk.resume_epoch,
                  callbacks=[tensorboard_cbk, model_checkpoint_cbk] + val_cbk)
        
        print("MODEL SUMMARY ", model.summary())

    elif cmd.mode == 'eval' or cmd.mode == 'validation':
        print("ENTERED VALIDATION")
        if cmd.mode=="validation":
            weights_dir = os.path.join(ckpt_dir,"train")
        else:
            weights_dir = os.path.join(ckpt_dir,"best")

        print("Evaluating on %s" % cmd.dataset)
        chosen_dataloader.get_dataset("eval", model_opts.dataloader_settings, batch_size=1)
        data = chosen_dataloader.dataset
        print("Got the dataset")
        tb_callback = tf.keras.callbacks.TensorBoard(log_dir=cmd.log_dir, profile_batch='10, 25')

        model = JointM4DepthSemantic(nbre_levels=nbre_levels, ablation_settings=model_opts.ablation_settings, num_classes = chosen_dataloader.class_count)
        print("Got the Model")
        #model_checkpoint_cbk = CustomCheckpointCallback(weights_dir, resume_training=True)
        model_checkpoint_cbk = CustomCheckpointCallback(weights_dir)
        #model.compile(metrics=[AbsRelError(),
                               #SqRelError(),
                               #RootMeanSquaredError(),
                               #RootMeanSquaredLogError(),
                               #ThresholdRelError(1), ThresholdRelError(2), ThresholdRelError(3)])
                               
        
        '''
        model.compile(metrics = [[AbsRelError(),
                               SqRelError(),
                               RootMeanSquaredError(),
                               RootMeanSquaredLogError(),
                               ThresholdRelError(1), ThresholdRelError(2), ThresholdRelError(3)],[tf.keras.metrics.MeanIoU(num_classes = chosen_dataloader.class_count)]])
        '''
        model.compile()
        start = time.time()
        print("Compiled the Model")
        metrics = model.evaluate(data, callbacks=[model_checkpoint_cbk])
        end = time.time()
        print("EXECUTION TIME = ", end-start)
        print("METRICS: ",metrics)
        # Keep track of the computed performance
        if cmd.mode == 'validation':
            manager = BestCheckpointManager(os.path.join(ckpt_dir,"train"), os.path.join(ckpt_dir,"best"), keep_top_n=cmd.keep_top_n)
            perfs = {"abs_rel": [metrics[0]], "sq_rel": [metrics[1]], "rmse": [metrics[2]], "rmsel": [metrics[3]],
                     "a1": [metrics[4]], "a2": [metrics[5]], "a3": [metrics[6]], "miou": [metrics[7]]}
            manager.update_backup(perfs)
            string = ''
            for perf in metrics:
                string += format(perf, '.4f') + "\t\t"
            with open(os.path.join(*[ckpt_dir, "validation-perfs.txt"]), 'a') as file:
                file.write(string + '\n')
        else:
            #np.savetxt(os.path.join(*[ckpt_dir, "perfs-" + cmd.dataset + ".txt"]), metrics, fmt='%.18e', delimiter='\t',
                       #newline='\n')
            np.savetxt(os.path.join(*[ckpt_dir, "perfs-" + cmd.dataset + ".txt"]), metrics, delimiter='\t',
                       newline='\n')
            #print("no saving")
            #f = open(os.path.join(*[ckpt_dir, "perfs-" + cmd.dataset + ".txt"]))
            #f.write(metrics)
            #f.close
            #np.savetxt(os.path.join(*[ckpt_dir, "perfs-" + cmd.dataset + ".txt"]), metrics, fmt='%.18e', delimiter='\t',
                       #newline='\n')

    elif cmd.mode == "predict":
        chosen_dataloader.get_dataset("predict", model_opts.dataloader_settings, batch_size=1)
        data = chosen_dataloader.dataset

        model = JointM4DepthSemantic(nbre_levels=nbre_levels, ablation_settings=model_opts.ablation_settings, num_classes = chosen_dataloader.class_count)
        model.compile()
        model_checkpoint_cbk = CustomCheckpointCallback(os.path.join(ckpt_dir, "best"), resume_training=True)
        i=0
        first_sample = data.take(1)
        
        #print(first_sample)
        print("Predicting First Sample")
        model.predict(first_sample, callbacks=[model_checkpoint_cbk])
        print("MODEL SUMMARY ", model.summary())
        is_first_run = True
        
        class_index = chosen_dataloader.class_index
        start = time.time()
        #print("START TIME!!!!!!")
        # Do what you want with the outputs
        for i, sample in enumerate(data):
            #print(i)
            if not is_first_run and sample["new_traj"]:
                print("End of trajectory")

            is_first_run = False

            est = model([[sample], sample["camera"]]) # Run network to get estimates
            '''
            #d_est = est["depth"][0, :, :, :]        # Estimate : [h,w,1] matrix with depth in meter
            #d_gt = sample['depth'][0, :, :, :]      # Ground truth : [h,w,1] matrix with depth in meter
            s_est = est["semantic"][0, :, :, :]        # Estimate : [h,w,1] matrix with depth in meter
            seg_est = tf.math.argmax(s_est, 2)
            seg_gt = sample['semantic'][0, :, :, :]      # Ground truth : [h,w,1] matrix with depth in meter
            #i_rgb = sample['RGB_im'][0, :, :, :]    # RGB image : [h,w,3] matrix with rgb channels ranging between 0 and 1
            
            #d_est = np.array(d_est)
            #d_gt = np.array(d_gt)
            seg_est = np.array(seg_est)
            seg_gt = np.array(seg_gt)
            #i_rgb = np.array(i_rgb)
            
            '''
            #d_gt[ d_gt > 255] = 255
            #d_gt[ d_gt < 0] = 0
            #d_est[ d_est > 255] = 255
            #d_est[ d_est < 0] = 0
            #d_est_rgb = np.squeeze(d_est, axis=2).astype(np.uint8)
            #d_gt_rgb = np.squeeze(d_gt, axis=2).astype(np.uint8)
            #im = Image.fromarray(d_est_rgb)
            #im.save("/media/DATA_4TB/Yara/results/depth_estimation/depth_"+str(i)+".png")
            #im = Image.fromarray(d_gt_rgb)
            #im.save("/media/DATA_4TB/Yara/results/gt_depth/depth_"+str(i)+".png")
            
            '''
            seg_est = np.expand_dims(seg_est, axis = 2)
            x1 = np.copy(seg_est)
            x2 = np.copy(seg_est)
            x3 = np.copy(seg_est)
            
            for key in class_index:
                print(key)
                x1[x1 == key] = class_index[key][0][0]
                x2[x2 == key] = class_index[key][0][1]
                x3[x3 == key] = class_index[key][0][2]
                
            img_seg = np.append(x1,x2, axis = 2)
            img_seg = np.append(img_seg,x3, axis = 2)
            
            im = Image.fromarray(img_seg.astype(np.uint8))
            '''
            '''
            im = Image.fromarray(seg_est.astype(np.uint8))
            im.save("/media/DATA_4TB/Yara/results/seg_estimation/sem_"+str(i)+".png")
            '''
            '''
            x1 = np.copy(seg_gt)
            x2 = np.copy(seg_gt)
            x3 = np.copy(seg_gt)
            
            for key in class_index:
                x1[x1 == key] = class_index[key][0][0]
                x2[x2 == key] = class_index[key][0][1]
                x3[x3 == key] = class_index[key][0][2]
                
            img_seg = np.append(x1,x2, axis = 2)
            img_seg = np.append(img_seg,x3, axis = 2)
            
            #print(img_seg)
            #print(np.shape(img_seg))
            im = Image.fromarray(img_seg.astype(np.uint8))
            '''
            '''
            seg_gt = np.squeeze(seg_gt)
            im = Image.fromarray(seg_gt.astype(np.uint8))
            im.save("/media/DATA_4TB/Yara/results/gt_sem/sem_"+str(i)+".png")
            '''
            '''
            i_rgb_rgb =  ( i_rgb  * 255.0).astype(np.uint8)
            im = Image.fromarray(i_rgb_rgb)
            im.save("/media/DATA_4TB/Yara/results/images/img_"+str(i)+".png")
            
            '''

        end = time.time()
        print("END TIME!!!!!!")
        print(end-start)
