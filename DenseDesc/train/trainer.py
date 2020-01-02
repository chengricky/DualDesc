import os
import yaml
import time

from skimage.io import imread
from torch.nn import DataParallel
from torch.utils.data import DataLoader
import torch.nn.functional as F

from DenseDesc.dataset.correspondence_database import CorrespondenceDatabase
from DenseDesc.match.match_utils import keep_valid_kps, compute_matching_score, detect_and_compute_sift
from DenseDesc.network.loss import interpolate_feats, normalize_coordinates
from DenseDesc.network.network_utils import l2_normalize
from DenseDesc.utils.config import cfg as path_cfg

from DenseDesc.dataset.correspondence_dataset import CorrespondenceDataset, detect_dog_keypoints, gray_repeats, normalize_image
from DenseDesc.network.wrapper import FeatureNetworkWrapper
from DenseDesc.train.train_tools import *


class Trainer:
    def _init_config(self):
        with open(os.path.join(path_cfg.project_dir, 'train_default.yaml'), 'r') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        self.name = self.config['name']
        with open(os.path.join(path_cfg.project_dir, 'dataset_default.yaml'), 'r') as f:
            self.dataset_config = yaml.load(f, Loader=yaml.FullLoader)
        self.recorder = Recorder(os.path.join('DenseDesc', 'data', 'record', self.name),
                                 os.path.join('DenseDesc', 'data', 'record', self.name + '.log'))
        self.model_dir = os.path.join('DenseDesc', 'data', 'model', self.name)
        print('==>Loaded Configuration File.')

    def _init_network(self):
        self.network = FeatureNetworkWrapper(cfg=self.config)
        self.network = DataParallel(self.network).cuda()
        self.extractor = self.network.module.extractor
        self.optim = torch.optim.Adam(self.extractor.parameters())
        self.model_dir = os.path.join('DenseDesc', 'data', 'model', self.name)
        self.epoch = 0
        self._load_model(self.model_dir, -1, True, True)
        print('==>Loaded Network Model.')

    def _init_dataset(self):
        self.database = CorrespondenceDatabase()
        train_set = []
        for name in self.config['trainset']:
            train_set += self.database.__getattribute__(name + "_set")
        print('==>database ok!')
        self.train_set = CorrespondenceDataset(self.dataset_config, train_set, self.database.background_pths)
        print('==>dataset ok!')
        self.train_set = DataLoader(self.train_set, self.config['batch_size'],
                                    True, num_workers=self.config['num_workers'])
        print('==>Loaded Dataset.')

    def __init__(self):
        self._init_config()
        self._init_network()
        self._init_dataset()

    def _save_model(self):
        os.system('mkdir -p {}'.format(self.model_dir))
        state_dict = {
            'extractor': self.extractor.state_dict(),
            'optim': self.optim.state_dict(),
            'epoch': self.epoch
        }
        torch.save(state_dict, os.path.join(self.model_dir, '{}.pth'.format(self.epoch)))

    def _load_model(self, model_dir, epoch=-1, load_extractor=True, load_optimizer=True):
        if not os.path.exists(model_dir):
            print('no model exists in {} !'.format(model_dir))
            return 0

        pths = [int(pth.split('.')[0]) for pth in os.listdir(model_dir)]
        if len(pths) == 0:
            print('no model exists in {} !'.format(model_dir))
            return 0
        if epoch == -1:
            pth = max(pths)
        else:
            pth = epoch

        pretrained_model = torch.load(os.path.join(model_dir, '{}.pth'.format(pth)))
        if load_extractor:
            self.extractor.load_state_dict(pretrained_model['extractor'])
        if load_optimizer:
            self.optim.load_state_dict(pretrained_model['optim'])
        print('load {} epoch {}'.format(model_dir, pretrained_model['epoch']))
        self.epoch = pretrained_model['epoch'] + 1

    def _get_warm_up_lr(self):
        if self.epoch <= 10:
            lr = 1e-4 * (self.epoch + 1)
        elif self.epoch <= 20:
            lr = 1e-3
        else:  # self.epoch<50:
            lr = max(1e-4 * (10 - (self.epoch - 20) / 3), 1e-4)
        return lr

    def train_model(self):
        reset_learning_rate(self.optim, self._get_warm_up_lr())
        begin_epoch = self.epoch
        for epoch in range(begin_epoch, self.config['epoch_num']):
            self.test_hpatch_match(self.database.hv_set, 'view-hp', 'dog')
            self.train_epoch()
            self._save_model()
            self.epoch += 1

    def _get_hem_thresh(self):
        hem_thresh = max(self.config['hem_thresh_begin'] - self.epoch * self.config['hem_thresh_decay_rate'],
                         self.config['hem_thresh_end'])
        print('current hem thresh {}'.format(hem_thresh))
        return hem_thresh

    def train_epoch(self):
        self.network.train()
        train_begin = time.time()
        batch_begin = time.time()
        hem_thresh = self._get_hem_thresh()
        for step, data in enumerate(self.train_set):
            loss_info = OrderedDict()

            img0, img1, pix_pos0, pix_pos1 = [item.cuda() for item in data]
            data_time = time.time() - batch_begin

            results = self.network(img0, img1, pix_pos0, pix_pos1, hem_thresh)
            loss = 0.0
            for k, v in results.items():
                v = torch.mean(v)
                if k.endswith('loss'): loss = loss + v
                loss_info[k] = v.cpu().detach().numpy()

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            batch_time = time.time() - batch_begin
            total_step = self.epoch * self.config['epoch_steps'] + step
            loss_info['loss'] = loss.cpu().detach().numpy()
            loss_info['data_time'] = data_time
            loss_info['batch_time'] = batch_time
            self.recorder.rec_loss(loss_info, total_step, self.epoch, 'train',
                                   ((total_step + 1) % self.config['record_step']) == 0)
            batch_begin = time.time()
            if step > self.config['epoch_steps']: break

        print('train epoch {} cost {} s'.format(self.epoch, time.time() - train_begin))

    def extract_feats(self, img, kps):
        # resize img to suitable size
        kps = kps.copy().astype(np.float32)
        kps = torch.tensor(kps, dtype=torch.float32)[None, :, :]
        img = gray_repeats(img)
        img = normalize_image(img)[None, :, :, :]

        self.network.eval()
        with torch.no_grad():
            feat_maps = self.extractor(img.cuda())
            # print(feat_maps.shape,kps.shape)
            pool_num = img.shape[-1] // feat_maps.shape[-1]
            kps = (kps + 0.5) / pool_num - 0.5
            feats=self._interpolate_feats(feat_maps,kps.cuda())[0]
            feats = l2_normalize(feats)

        return feats.cpu().numpy()

    ########################### hpatch evaluation ###############################
    @staticmethod
    def _get_feats_kps(pth, model_name):
        if model_name == 'sift':
            kps, feats = detect_and_compute_sift(imread(pth))
            kps = np.asarray([kp.pt for kp in kps])
        else:
            subpths = pth.split('/')
            npzfn = '_'.join([subpths[-2], subpths[-1].split('.')[0]]) + '.npz'
            data_dir = subpths[-3]
            fn = os.path.join('DenseDesc', 'data', model_name, data_dir)
            fn = os.path.join(fn, npzfn)
            if os.path.exists(fn):
                npzfile = np.load(fn)
                kps, feats = npzfile['kpts'], npzfile['descs']
            else:
                kps, feats= np.zeros([0,2]), np.zeros([0,128])
                print('{} not found !'.format(fn))
        return kps, feats

    model_name_to_dir_name = {
        'superpoint': 'sp_hpatches',
        'geodesc': 'gd_hpatches',
        'geodesc_ds': 'gd_hpatches_ds',
        'sift':'sift',
        'lf_net':'lf_hpatches'
    }

    def _get_kpts(self, pth0, pth1, kpts_type):
        if kpts_type in self.model_name_to_dir_name:
            kps0, feats0 = self._get_feats_kps(pth0, self.model_name_to_dir_name[kpts_type])
            kps1, feats1 = self._get_feats_kps(pth1, self.model_name_to_dir_name[kpts_type])
        else:  # Use DoG
            img0, img1 = imread(pth0), imread(pth1)
            kps0, kps1 = detect_dog_keypoints(img0), detect_dog_keypoints(img1)

        return kps0, kps1

    def test_hpatch_match(self, database, prefix, kpts_type):

        self.network.eval()
        for idx, data in enumerate(database):
            pth0, pth1, H = data['img0_pth'], data['img1_pth'], data['H'].copy()
            img0, img1 = imread(pth0), imread(pth1)
            kps0, kps1 = self._get_kpts(pth0, pth1, kpts_type)

            kps0 = keep_valid_kps(kps0, H, img1.shape[0], img1.shape[1], flow=None)
            kps1 = keep_valid_kps(kps1, np.linalg.inv(H), img0.shape[0], img0.shape[1], flow=None)

            if len(kps0) == 0 or len(kps1) == 0:
                ratio, number = 0.0, 0
            else:
                feats0 = self.extract_feats(img0, kps0)
                feats1 = self.extract_feats(img1, kps1)
                ratio, number = compute_matching_score(feats0, kps0, feats1, kps1, H, self.config['match_thresh'])

            loss_info = OrderedDict()
            loss_info['match_ratio'] = ratio
            loss_info['match_number'] = number
            self.recorder.rec_loss(loss_info, self.epoch, self.epoch, '{}_test_match'.format(prefix), False)
        self.recorder.rec_loss({}, self.epoch, self.epoch, '{}_test_match'.format(prefix), True)

    @staticmethod
    def _interpolate_feats(feats,pos):
        with torch.no_grad():
            pos_norm = normalize_coordinates(pos, feats.shape[2], feats.shape[3])
            pos_norm = torch.unsqueeze(pos_norm, 1)  # b,1,n,2

        feats_pos = F.grid_sample(feats, pos_norm, 'bilinear', 'border')[:, :, 0, :]  # b,f,n
        feats_pos = feats_pos.permute(0, 2, 1)  # b,n,f
        return feats_pos


