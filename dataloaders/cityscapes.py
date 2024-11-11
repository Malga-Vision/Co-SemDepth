import tensorflow as tf
import numpy as np
from .generic import *

class DataLoaderCityScapes(DataLoaderGeneric):
    """Dataloader for the Cityscapes dataset
    """
    def __init__(self, out_size=[384,768], crop=False):
        super(DataLoaderCityScapes, self).__init__('cityscapes')

        self.in_size = [1024, 2048]
        self.class_count = 20
        self.class_index = {
        0: [(128, 64, 128),  'road'],
        1: [(244, 35, 232),  'sidewalk'],
        2: [(70, 70, 70),  'building'],
        3: [(102, 102, 156),  'wall'],
        4: [(190, 153, 153),  'fence'],
        5: [(153, 153, 153),  'pole'],
        6: [(250, 170, 30),  'traffic light'],
        7: [(220, 220, 0),  'traffic sign'],
        8: [(107, 142, 35),  'vegetation'],
        9: [(152, 251, 152),  'terrain'],
        10: [(70, 130, 180),  'sky'],
        11: [(220, 20, 60),  'person'],
        12: [(255, 0, 0),  'rider'],
        13: [(0, 0, 142),  'car'],
        14: [(0, 0, 70),  'truck'],
        15: [(0, 60, 100),  'bus'],
        16: [(0, 80, 100),  'train'],
        17: [(0, 0, 230),  'motorcycle'],
        18: [(119, 11, 32),  'bicycle'],
        19: [(0, 0, 0),  'void']
        }
        """
        self.class_index = {
        0: [19 ,(0, 0, 0),  'void'],
        1: [19, (0, 0, 0),  'void'],
        2: [19, (0, 0, 0),  'void'],
        3: [19, (0, 0, 0),  'void'],
        4: [19, (0, 0, 0),  'void'],
        5: [19, (0, 0, 0),  'void'],
        6: [19, (0, 0, 0),  'void'],
        7: [0, (128, 64, 128),  'road'],
        8: [1, (244, 35, 232),  'sidewalk'],
        9: [19, (0, 0, 0),  'void'],
        10: [19, (0, 0, 0),  'void'],
        11: [2, (70, 70, 70),  'building'],
        12: [3, (102, 102, 156),  'wall'],
        13: [4, (190, 153, 153),  'fence'],
        14: [19, (0, 0, 0),  'void'],
        15: [19, (0, 0, 0),  'void'],
        16: [19, (0, 0, 0),  'void'],
        17: [5, (153, 153, 153),  'pole'],
        18: [19, (0, 0, 0),  'void'],
        19: [6, (250, 170, 30),  'traffic light'],
        20: [7, (220, 220, 0),  'traffic sign'],
        21: [8, (107, 142, 35),  'vegetation'],
        22: [9, (152, 251, 152),  'terrain'],
        23: [10, (70, 130, 180),  'sky'],
        24: [11, (220, 20, 60),  'person'],
        25: [12, (255, 0, 0),  'rider'],
        26: [13, (0, 0, 142),  'car'],
        27: [14, (0, 0, 70),  'truck'],
        28: [15, (0, 60, 100),  'bus'],
        29: [19, (0, 0, 0),  'void'],
        30: [19, (0, 0, 0),  'void'],
        31: [16, (0, 80, 100),  'train'],
        32: [17, (0, 0, 230),  'motorcycle'],
        33: [18, (119, 11, 32),  'bicycle'],
        -1: [19, (0, 0, 0),  'void']
        }
        """

    def _set_output_size(self, out_size=[384, 768]):
        self.out_size = out_size
        self.long_edge = 0 if out_size[0]>=out_size[1] else 1
        if self.crop:
            self.intermediate_size = [out_size[self.long_edge], out_size[self.long_edge]]
        else:
            self.intermediate_size = out_size
        self.fx = 1.1 * self.intermediate_size[1]
        self.fy = 2.19 * self.intermediate_size[0]
        self.cx = 0.525 * self.intermediate_size[1]
        self.cy = 0.502 * self.intermediate_size[0]

    def get_dataset(self, usecase, settings, batch_size=3, out_size=[384, 768], crop=False):
        self.crop = crop
        if (usecase == "eval" or usecase=="predict") and self.crop:
            return AttributeError("Crop option should be disabled when evaluating or predicting samples")
        super(DataLoaderCityScapes, self).get_dataset(usecase, settings, batch_size=batch_size, out_size=out_size)

    @tf.function
    def _decode_samples(self, data_sample):
        file = tf.io.read_file(tf.strings.join([self.db_path, data_sample['camera_l']], separator='/'))
        image = tf.io.decode_png(file)
        rgb_image = tf.cast(image, dtype=tf.float32)/255.

        camera_data = {
            "f": tf.convert_to_tensor([self.fx, self.fy]),
            "c": tf.convert_to_tensor([self.cx, self.cy]),
        }
        out_data = {}
        out_data["camera"] = camera_data.copy()
        out_data['RGB_im'] = tf.reshape(tf.image.resize(rgb_image, self.intermediate_size), self.intermediate_size+[3])
        out_data['rot'] = tf.cast(tf.stack([data_sample['qw'],data_sample['qx'],data_sample['qy'],data_sample['qz']], 0), dtype=tf.float32)
        out_data['trans'] = tf.cast(tf.stack([data_sample['tx'],data_sample['ty'],data_sample['tz']], 0), dtype=tf.float32)
        out_data['new_traj'] = tf.math.equal(data_sample['id'], 0)
        
        # Load semantic data only if they are available
        if 'semantic' in data_sample:
           file = tf.io.read_file(tf.strings.join([self.db_path, data_sample['semantic']], separator='/'))
           image = tf.image.decode_png(file)
           
           out_data['semantic'] = tf.reshape(tf.image.resize(image, self.intermediate_size, method = "nearest"), self.intermediate_size + [1])
           
           
        return out_data

    def _perform_augmentation(self):
        # flip and transpose image

        if not self.usecase == "finetune":
            self._augmentation_step_flip()

            # we can transpose h and w dimensions if images have a square shape as a data augmentation
            if self.intermediate_size[0] == self.intermediate_size[1]:
                im_col = self.out_data["RGB_im"]
                #im_depth = self.out_data["depth"]
                im_semantic = self.out_data["semantic"]
                rot = self.out_data["rot"]
                trans = self.out_data["trans"]

                def do_nothing():
                    #return [im_col, im_depth, rot, trans]
                    return [im_col, im_semantic, rot, trans]

                def true_transpose():
                    col = tf.transpose(im_col, perm=[0, 2, 1, 3])
                    #dep = tf.transpose(im_depth, perm=[0, 2, 1, 3])
                    semantic = tf.transpose(im_semantic, perm=[0, 2, 1, 3])
                    r = tf.stack([rot[:, 0], -rot[:, 2], -rot[:, 1], -rot[:, 3]], axis=1)
                    t = tf.stack([trans[:, 1], trans[:, 0], trans[:, 2]], axis=1)
                    #return [col, dep, r, t]
                    return [col, semantic, r, t]

                p_order = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
                pred = tf.less(p_order, 0.5)
                #im_col, im_depth, rot, trans = tf.cond(pred, true_transpose, do_nothing)
                im_col, im_semantic, rot, trans = tf.cond(pred, true_transpose, do_nothing)

                #self.out_data["depth"] = im_depth
                self.out_data["semantic"] = im_semantic
                self.out_data["RGB_im"] = im_col
                self.out_data["rot"] = rot
                self.out_data["trans"] = trans

        # crop image to the desired output size
        if self.crop:
            if self.long_edge == 0:
                diff = self.intermediate_size[1]-self.out_size[1]
                offset = tf.random.uniform(shape=[], minval=0, maxval=diff, dtype=tf.int32)
                self.out_data['RGB_im'] = tf.slice(self.out_data['RGB_im'], [0, 0, offset, 0], [self.seq_len, self.out_size[0], self.out_size[1], 3])
                #self.out_data['depth'] = tf.slice(self.out_data['depth'], [0, 0, offset, 0], [self.seq_len, self.out_size[0], self.out_size[1], 1])
                self.out_data['semantic'] = tf.slice(self.out_data['semantic'], [0, 0, offset, 0], [self.seq_len, self.out_size[0], self.out_size[1], 1])
                self.out_data['camera']['c'] = tf.convert_to_tensor([self.out_data['camera']['c'][0]-tf.cast(offset, tf.float32), self.out_data['camera']['c'][1]])
            else:
                diff = self.intermediate_size[0]-self.out_size[0]
                offset = tf.random.uniform(shape=[], minval=0, maxval=diff, dtype=tf.int32)
                self.out_data['RGB_im'] = tf.slice(self.out_data['RGB_im'], [0, offset, 0, 0], [self.seq_len, self.out_size[0],  self.out_size[1], 3])
                #self.out_data['depth'] = tf.slice(self.out_data['depth'], [0, offset, 0, 0], [self.seq_len, self.out_size[0], self.out_size[1], 1])
                self.out_data['semantic'] = tf.slice(self.out_data['semantic'], [0, offset, 0, 0], [self.seq_len, self.out_size[0], self.out_size[1], 1])
                self.out_data['camera']['c'] = tf.convert_to_tensor([self.out_data['camera']['c'][0], self.out_data['camera']['c'][1]-tf.cast(offset, tf.float32)])
            self.out_data['RGB_im'] = tf.reshape(self.out_data['RGB_im'], [self.seq_len, self.out_size[0],  self.out_size[1], 3])
            #self.out_data['depth'] = tf.reshape(self.out_data['depth'], [self.seq_len, self.out_size[0],  self.out_size[1], 1])
            self.out_data['semantic'] = tf.reshape(self.out_data['semantic'], [self.seq_len, self.out_size[0],  self.out_size[1], 1])


        self._augmentation_step_color()
