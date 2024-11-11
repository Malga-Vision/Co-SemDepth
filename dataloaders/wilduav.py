import tensorflow as tf
from .generic import *

class DataLoaderWUAV(DataLoaderGeneric):
    """Dataloader for the WildUAV dataset
    """
    def __init__(self, out_size=[384, 528]):
        super(DataLoaderWUAV, self).__init__('wilduav')

        self.in_size = [3956, 5280]
        self.depth_type = "map"
        self.class_count = 9
        self.class_index = {
        0: [(127, 175, 230), 'Sky'],
        1: [(75, 163, 185),'water plane'],
        2: [(50, 128, 0),  'Trees'],
        3: [(53, 94, 59),  'land'],
        4: [(237, 125, 49), 'vehicle'],   
        5: [(70, 70, 70),   'boulders'],
        6: [(128, 64, 128),  'Road'],
        7: [(64, 64, 128),  'man-made construction'],
        8: [(128, 64, 64),  'others']
        }
        self.class_index_v0 = {
        0: [(127, 175, 230), 'Sky'],
        1: [(75, 163, 185),'water plane'],
        2: [(50, 128, 0),  'Trees'],
        3: [(117, 97, 97),  'Dirt Ground'],
        4: [(53, 94, 59), 'Ground vegetation'],   
        5: [(70, 70, 70),   'Rocks'],
        6: [(128, 64, 128),  'Road'],
        7: [(64, 64, 128),  'man-made construction'],
        8: [(128, 64, 64),  'others']
        }
        """
        
        self.class_index = {
        0: [[0, 255, 255], 'Sky'],
        1: [ [0, 127, 0], 'Deciduous trees'],
        2: [[19, 132, 69], 'Coniferous trees'],
        3: [[0, 53, 65], 'Fallen trees'],
        4: [[130, 76, 0], 'Dirt ground'],
        5: [[152, 251, 152], 'Ground vegetation'],
        6: [[151, 126, 171], 'Rocks'],
        7: [[0, 0, 255], 'Water plane'],
        8: [[250, 150, 0], 'Building'],
        9: [[115, 176, 195], 'Fence'],
        10: [[128, 64, 128], 'Road'],
        11: [[255, 77, 228], 'Sidewalk'],
        12: [[123, 123, 123], 'Static car'],
        13: [[255, 255, 255], 'Moving car'],
        14: [[200, 0, 0], 'People'],
        15: [[0, 0, 0], 'Empty']
        }
        """

    def _set_output_size(self, out_size=[384, 528]):
        self.out_size = out_size
        self.fx = 0.862 * self.out_size[1]
        self.fy = 1.1503 * self.out_size[0]
        self.cx = 0.5012 * self.out_size[1]
        self.cy = 0.497 * self.out_size[0]
        
    def get_dataset(self, usecase, settings, batch_size=3, out_size=[384, 528], crop=False):
        self.crop = crop
        if (usecase == "eval" or usecase=="predict") and self.crop:
            return AttributeError("Crop option should be disabled when evaluating or predicting samples")
        super(DataLoaderWUAV, self).get_dataset(usecase, settings, batch_size=batch_size, out_size=out_size)
        
    def _decode_samples(self, data_sample):
        file = tf.io.read_file(tf.strings.join([self.db_path, data_sample['camera_l']], separator='/'))
        image = tf.io.decode_png(file)
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
        
        if 'depth' in data_sample:
            sem_path = tf.strings.regex_replace(data_sample['depth'], "depth","seg_upd")
            sem_path = tf.strings.regex_replace(sem_path, ".npy",".png")
            file = tf.io.read_file(tf.strings.join([self.db_path, sem_path], separator='/'))
            image = tf.image.decode_png(file)
            semantic = tf.reshape(image[:,:,0], self.in_size + [1])
            out_data['semantic'] = tf.image.resize(semantic, self.out_size, method = "nearest")
            
            
            #im_greyscale = tf.math.reduce_euclidean_norm(out_data['RGB_im'], axis=-1, keepdims=True)
            #mask = tf.cast(tf.greater(im_greyscale, 0.), tf.float32)
            file = tf.io.read_file(tf.strings.join([self.db_path, data_sample['depth']], separator='/'))
            image = tf.io.decode_raw(file, tf.float32)
            image = image[-(self.in_size[0]*self.in_size[1]):] ## quite strange!
            depth = tf.reshape(tf.cast(image, dtype=tf.float32), self.in_size+[1])
            # WARNING we disable areas with no color information
            out_data['depth'] = tf.reshape(tf.image.resize(depth, self.out_size, method='nearest'), self.out_size+[1])#*mask

        return out_data

    def _perform_augmentation(self):
        self._augmentation_step_flip()
        self._augmentation_step_color()
