"""
Generate Image Paths for SUN / COCO / HPatches (image pairs with homography)
"""
import os
from DenseDesc.utils.config import cfg
import numpy as np
import pickle
from skimage.io import imread, imsave
import cv2


def read_pickle(pkl_path):
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)


def save_pickle(data, pkl_path):
    os.system('mkdir -p {}'.format(os.path.dirname(pkl_path)))
    with open(pkl_path, 'wb') as f:
        pickle.dump(data, f)


class CorrespondenceDatabase:
    @staticmethod
    def get_SUN397_image_paths():
        img_dir = os.path.join('/home/ricky/datasets', 'SUN397')
        img_pths = []
        with open(os.path.join(img_dir, 'ClassName.txt'), mode='r') as cls:
            line = cls.readline()
            while len(line) > 4:
                sub_dir = img_dir + line[:-1]
                img_pth = [os.path.join(sub_dir, fn) for fn in os.listdir(sub_dir)]
                img_pths += img_pth
                line = cls.readline()
        return img_pths

    @staticmethod
    def get_SUN2012_image_paths():
        img_dir = os.path.join(cfg.data_dir, 'SUN2012Images', 'JPEGImages')
        img_pths = [os.path.join(img_dir, fn) for fn in os.listdir(img_dir)]
        return img_pths

    @staticmethod
    def get_COCO_image_paths():
        img_dir = os.path.join(cfg.data_dir, 'coco', 'train2014')
        img_pths = [os.path.join(img_dir, fn) for fn in os.listdir(img_dir)]
        img_dir = os.path.join(cfg.data_dir, 'coco', 'val2014')
        img_pths += [os.path.join(img_dir, fn) for fn in os.listdir(img_dir)]
        return img_pths

    @staticmethod
    def get_hpatch_sequence_database(name='resize', max_size=480):
        """
        Get hpatches_resize if it exists, else generate one
        """
        def resize_and_save(pth_in, max_size, pth_out):
            img = imread(pth_in)
            h, w = img.shape[:2]
            ratio = max_size / max(h, w)
            h, w = int(h * ratio), int(w * ratio)
            img = cv2.resize(img, (w, h))
            imsave(pth_out, img)
            return ratio

        root_dir = os.path.join(cfg.data_dir, 'hpatches-sequences-release')
        output_dir = os.path.join(cfg.data_dir, 'hpatches_{}'.format(name))
        pkl_file = os.path.join(output_dir, 'info.pkl')
        if os.path.exists(pkl_file):
            return read_pickle(pkl_file)

        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        illumination_dataset = []
        viewpoint_dataset = []
        for dir in os.listdir(root_dir):
            if not os.path.exists(os.path.join(output_dir, dir)):
                os.mkdir(os.path.join(output_dir, dir))

            img_pattern = os.path.join(root_dir, dir, '{}.ppm')
            hmg_pattern = os.path.join(root_dir, dir, 'H_1_{}')
            omg_pattern = os.path.join(output_dir, dir, '{}.png')

            ratio0 = resize_and_save(img_pattern.format(1),max_size,omg_pattern.format(1))
            # resize image
            for k in range(2,7):
                ratio1=resize_and_save(img_pattern.format(k),max_size,omg_pattern.format(k))
                H = np.loadtxt(hmg_pattern.format(k))
                H = np.matmul(np.diag([ratio1, ratio1, 1.0]), np.matmul(H, np.diag([1 / ratio0, 1 / ratio0, 1.0])))
                data = {'type': 'hpatch',
                       'img0_pth': omg_pattern.format(1),
                       'img1_pth': omg_pattern.format(k),
                       'H': H}
                if dir.startswith('v'):
                    viewpoint_dataset.append(data)
                if dir.startswith('i'):
                    illumination_dataset.append(data)

        save_pickle([illumination_dataset, viewpoint_dataset], pkl_file)
        return illumination_dataset, viewpoint_dataset

    @staticmethod
    def generate_homography_database(img_list):
        return [{'type': 'homography', 'img_pth': img_pth} for img_pth in img_list]

    def __init__(self):
        # self.coco_set=self.generate_homography_database(self.get_COCO_image_paths())
        # print('coco len {}'.format(len(self.coco_set)))

        self.sun_set = self.generate_homography_database(self.get_SUN397_image_paths())
        print('sun len {}'.format(len(self.sun_set)))

        self.hi_set, self.hv_set = self.get_hpatch_sequence_database()

        # img_dir = os.path.join(cfg.data_dir, 'SUN2012Images', 'JPEGImages')
        # self.background_pths = [os.path.join(img_dir, fn) for fn in os.listdir(img_dir)]
        self.background_pths = self.sun_set
