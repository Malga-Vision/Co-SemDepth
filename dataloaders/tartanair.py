import tensorflow as tf
from .generic import *
import numpy as np

class DataLoaderTartanAir(DataLoaderGeneric):
    """Dataloader for the TartanAir dataset
    """
    def __init__(self, out_size=[384, 512]):
        super(DataLoaderTartanAir, self).__init__('tartanair')

        self.in_size = [480, 640]
        self.depth_type = "map"
        self.class_count = 9
        self.class_index = {
        0: [(127, 175, 230), 'Sky'],
        1: [(75, 163, 185),'water plane'],
        2: [(50, 128, 0),  'Trees'],
        3: [(105, 58, 69),  'Dirt Ground'],
        4: [(53, 94, 59), 'Ground vegetation'],   
        5: [(117, 97, 97),   'Rocky ground'],
        6: [(128, 64, 128),  'Road'],
        7: [(64, 64, 128),  'man-made construction'],
        8: [(128, 64, 64),  'others']
        }

    def _set_output_size(self, out_size=[384, 512]):
        self.out_size = out_size
        self.fx = 0.5 * self.out_size[1]
        self.fy = 2./3. * self.out_size[0]
        self.cx = 0.5 * self.out_size[1]
        self.cy = 0.5 * self.out_size[0]

    def _decode_samples(self, data_sample):
        file = tf.io.read_file(tf.strings.join([self.db_path, data_sample['camera_l']], separator='/'))
        image = tf.io.decode_jpeg(file)
        rgb_image = tf.cast(image, dtype=tf.float32)/255.

        camera_data = {
            "f": tf.convert_to_tensor([self.fx, self.fy], dtype=tf.float32),
            "c": tf.convert_to_tensor([self.cx, self.cy], dtype=tf.float32),
        }
        out_data = {}
        out_data["camera"] = camera_data.copy()
        out_data['RGB_im'] = tf.reshape(tf.image.resize(rgb_image, self.out_size), self.out_size+[3])
        out_data['rot'] = tf.cast(tf.stack([data_sample['qw'],data_sample['qx'],data_sample['qy'],data_sample['qz']], 0), dtype=tf.float32)
        out_data['trans'] = tf.cast(tf.stack([data_sample['tx'],data_sample['ty'],data_sample['tz']], 0), dtype=tf.float32)
        out_data['new_traj'] = tf.math.equal(data_sample['id'], 0)

        # Load depth data only if they are available
        if 'depth' in data_sample:
            #im_greyscale = tf.math.reduce_euclidean_norm(out_data['RGB_im'], axis=-1, keepdims=True)
            #mask = tf.cast(tf.greater(im_greyscale, 0.), tf.float32)
            #sem_path = data_sample['depth'].replace("depth_left","seg_upd")
            #sem_path = sem_path.replace("depth.npy","seg.npy")
            sem_path = tf.strings.regex_replace(data_sample['depth'], "depth_left","seg_upd")
            sem_path = tf.strings.regex_replace(sem_path, "depth.npy","seg.npy")
            file = tf.io.read_file(tf.strings.join([self.db_path, sem_path], separator='/'))
            image = tf.io.decode_raw(file, tf.uint8)
            image = image[-(self.in_size[0]*self.in_size[1]):]
            semantic = tf.reshape(image, self.in_size + [1])
            out_data['semantic'] = tf.image.resize(semantic, self.out_size, method = "nearest")
            
            im_greyscale = tf.math.reduce_euclidean_norm(out_data['RGB_im'], axis=-1, keepdims=True)
            mask = tf.cast(tf.greater(im_greyscale, 0.), tf.float32)
            file = tf.io.read_file(tf.strings.join([self.db_path, data_sample['depth']], separator='/'))
            image = tf.io.decode_raw(file, tf.float32)
            image = image[-(self.in_size[0]*self.in_size[1]):]
            depth = tf.reshape(tf.cast(image, dtype=tf.float32), self.in_size+[1])
            # WARNING we disable areas with no color information
            out_data['depth'] = tf.reshape(tf.image.resize(depth, self.out_size, method='nearest'), self.out_size+[1])*mask
            #
            #
            #image = image[-(self.in_size[0]*self.in_size[1]):]
            #semantic = tf.reshape(tf.cast(image, dtype=tf.float32), self.in_size+[1])
            # WARNING we disable areas with no color information
            #out_data['depth'] = tf.reshape(tf.image.resize(depth, self.out_size, method='nearest'), self.out_size+[1])*mask
            

        return out_data

    def _perform_augmentation(self):
        self._augmentation_step_flip()
        self._augmentation_step_color()
