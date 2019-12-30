import torch

from dataset.augmentation_utils import *
from dataset.homography import *
from torch.utils.data import Dataset
from torchvision.transforms import ColorJitter

from skimage.io import imread

from match.match_utils import detect_dog_keypoints, perspective_transform


def normalize_image(img):
    img=(img.transpose([2,0,1]).astype(np.float32)-127.0)/128.0
    return torch.tensor(img,dtype=torch.float32)

def gray_repeats(img_raw):
    if len(img_raw.shape) == 2: img_raw = np.repeat(img_raw[:, :, None], 3, axis=2)
    if img_raw.shape[2] > 3: img_raw = img_raw[:, :, :3]
    return img_raw


def round_coordinates(coord,h,w):
    coord=np.round(coord).astype(np.int32)
    coord[coord[:,0]<0,0]=0
    coord[coord[:,0]>=w,0]=w-1
    coord[coord[:,1]<0,1]=0
    coord[coord[:,1]>=h,1]=h-1
    return coord


class CorrespondenceDataset(Dataset):
    def __init__(self, decode_args, database, background_path_list=None):
        super(CorrespondenceDataset,self).__init__()
        self.database=database

        self.args=decode_args
        self.jitter = ColorJitter(self.args['brightness'], self.args['contrast'], self.args['saturation'], self.args['hue'])

        self.background_pths = background_path_list

        self.name2func = {
            'jpeg': lambda img_in: jpeg_compress(img_in, self.args['jpeg_low'], self.args['jpeg_high']),
            'blur': lambda img_in: gaussian_blur(img_in, self.args['blur_range']),
            'jitter': lambda img_in: np.asarray(self.jitter(Image.fromarray(img_in))),
            'noise': lambda img_in: add_noise(img_in),
            'none': lambda img_in: img_in,
            'sp_gaussian_noise': lambda img_in: additive_gaussian_noise(img_in,self.args['sp_gaussian_range']),
            'sp_speckle_noise': lambda img_in: additive_speckle_noise(img_in,self.args['sp_speckle_prob_range']),
            'sp_additive_shade': lambda img_in: additive_shade(img_in,self.args['sp_nb_ellipse'],self.
                                                               args['sp_transparency_range'],self.args['sp_kernel_size_range']),
            'sp_motion_blur': lambda img_in: motion_blur(img_in,self.args['sp_max_kernel_size']),
            'resize_blur': lambda img_in: resize_blur(img_in, self.args['resize_blur_min_ratio'])
        }

    def __len__(self):
        return len(self.database)

    def __getitem__(self, index):
        return self.decode(self.database[index])

    def decode_homography(self, data):
        img_raw = imread(data['img_pth'])
        th, tw = self.args['h'],self.args['w']
        # if img_raw is [h,w] repeats it to [h,w,3]
        img_raw = gray_repeats(img_raw)

        # sample part of input image
        if np.random.random()<0.8:
            img_raw,_,_=self.get_image_region(img_raw,th*np.random.uniform(0.8,1.2),tw*np.random.uniform(0.8,1.2))

        #
        H = self.generate_homography()

        # warp image
        img0=cv2.resize(img_raw,(tw,th),interpolation=cv2.INTER_LINEAR)          # [h,w,3]
        img1=cv2.warpPerspective(img0,H,(tw,th),flags=cv2.INTER_LINEAR)          # [h,w,3]

        # add random background
        if self.args["add_background"]:
            img1=self.add_homography_background(img1, data['img_pth'], H)
        else:
            # todo: (set background to 127, which is 0 after normalization, instead of 0, which is -127 after normalization?)
            img1=self.add_black_background(img1,H)

        # get ground truth
        pix_pos0, pix_pos1 = self.sample_ground_truth(img0,H)
        # scale_offset, rotate_offset = self.compute_scale_rotate_offset(H,pix_pos0)
        return img0, img1, pix_pos0, pix_pos1

    def decode(self, data):
        if data['type']=='homography':
            img0, img1, pix_pos0, pix_pos1 = self.decode_homography(data)
        else:
            raise NotImplementedError

        if data['type'] != 'hpatch' and data['type'] != 'webcam':
            img0 = self.augment(img0)
            img1 = self.augment(img1)

        # to tensor
        # if self.args['equalize_hist']:
        #     img0, img1 = equal_hist(img0), equal_hist(img1)

        img0=normalize_image(img0)
        img1=normalize_image(img1)
        pix_pos0=torch.tensor(pix_pos0,dtype=torch.float32)
        pix_pos1=torch.tensor(pix_pos1,dtype=torch.float32)
        # scale_offset=torch.tensor(scale_offset,dtype=torch.int32)
        # rotate_offset=torch.tensor(rotate_offset,dtype=torch.int32)
        # H=torch.tensor(H,dtype=torch.float32)

        return img0, img1, pix_pos0, pix_pos1, # H, scale_offset, rotate_offset

    def add_homography_background(self, img, cur_path, H):
        if self.background_pths is None: # if not add background
            return img
        bpth = self.generate_background_pth(cur_path)

        h, w, _ = img.shape
        bimg = cv2.resize(imread(bpth), (w, h))
        if len(bimg.shape) == 2: bimg = np.repeat(bimg[:, :, None], 3, axis=2)
        if bimg.shape[2] > 3: bimg = bimg[:, :, :3]
        msk_tgt = cv2.warpPerspective(np.ones([h, w], np.uint8), H, (w, h), flags=cv2.INTER_NEAREST).astype(np.bool)
        img[np.logical_not(msk_tgt)] = bimg[np.logical_not(msk_tgt)]
        return img

    @staticmethod
    def add_black_background(img, H):
        h, w, _ = img.shape
        msk_tgt = cv2.warpPerspective(np.ones([h, w], np.uint8), H, (w, h), flags=cv2.INTER_NEAREST).astype(np.bool)
        img[np.logical_not(msk_tgt)] = 127
        return img

    def get_homography_correspondence(self, h, w, th, tw, H):
        coords = [np.expand_dims(item, 2) for item in np.meshgrid(np.arange(w), np.arange(h))]
        coords = np.concatenate(coords, 2).astype(np.float32)
        if self.args['perturb']: coords += np.random.randint(0, self.args['perturb_max'] + 1, coords.shape)
        coords_target = cv2.perspectiveTransform(np.reshape(coords, [1, -1, 2]), H.astype(np.float32))
        coords_target = np.reshape(coords_target, [h, w, 2])

        source_mask = np.logical_and(np.logical_and(0 <= coords_target[:, :, 0], coords_target[:, :, 0] < tw ),
                                     np.logical_and(0 <= coords_target[:, :, 1], coords_target[:, :, 1] < th ))
        coords_target[np.logical_not(source_mask)] = 0

        return coords_target, source_mask

    def sample_correspondence(self, img, pix_pos, msk):
        h, w = img.shape[0], img.shape[1]
        val_msk = []
        if self.args['test_canny']:
            edge = cv2.Canny(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY),
                             self.args['canny_thresh0'], self.args['canny_thresh1'])
            edge_mask = edge > 0
            val_msk.append(edge_mask)
        if self.args['test_harris']:
            harris_img = cv2.cornerHarris(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32), 2, 3, 0.04)
            harris_msk = harris_img > np.percentile(harris_img.flatten(), self.args['harris_percentile'])
            val_msk.append(harris_msk)

        if self.args['test_dog']:
            kps_msk = np.zeros_like(msk)
            kps = detect_dog_keypoints(img)
            if len(kps) > 0:
                kps = round_coordinates(kps, h, w)
                kps_msk[kps[:, 1], kps[:, 0]] = True
            val_msk.append(kps_msk)

        if self.args['test_edge']:
            edge_thresh = self.args['edge_thresh']
            edge_msk = np.ones_like(msk)
            edge_msk[:edge_thresh, :] = False
            edge_msk[:, :edge_thresh] = False
            edge_msk[h - edge_thresh:h, :] = False
            edge_msk[:, w - edge_thresh:w] = False
            edge_msk=np.logical_and(edge_msk,np.logical_and(pix_pos[:,:,0]<w-edge_thresh, pix_pos[:,:,0]>edge_thresh))
            edge_msk=np.logical_and(edge_msk,np.logical_and(pix_pos[:,:,1]<h-edge_thresh, pix_pos[:,:,1]>edge_thresh))
            msk = np.logical_and(msk, edge_msk)

        if len(val_msk) == 0:
            val_msk = np.ones_like(msk)
        else:
            val_msk_out = np.zeros_like(msk)
            for item in val_msk:
                val_msk_out = np.logical_or(val_msk_out, item)
            val_msk = val_msk_out
        val_msk = np.logical_and(msk, val_msk)

        hs, ws = np.nonzero(val_msk)
        pos_num = len(hs)
        if pos_num == 0:
            if self.args['test_edge']:
                edge_thresh = self.args['edge_thresh']
            else:
                edge_thresh = 0
            hs, ws = np.random.randint(edge_thresh, h - edge_thresh, 3000), np.random.randint(edge_thresh, w - edge_thresh, 3000)
            pos_num = len(hs)

        sample_num = self.args['sample_num']
        if pos_num >= sample_num:
            idxs = np.arange(pos_num)
            np.random.shuffle(idxs)
            idxs = idxs[:sample_num]
        else:
            idxs = np.arange(pos_num)
            idxs = np.append(idxs, np.random.choice(idxs, sample_num - pos_num))

        pix_pos0 = np.concatenate([ws[idxs][:, None], hs[idxs][:, None]], 1)  # sn,2
        pix_pos1 = pix_pos[pix_pos0[:, 1], pix_pos0[:, 0]]
        return pix_pos0, pix_pos1

    def sample_ground_truth(self, img, H, img_tgt=None):
        h,w,_=img.shape
        if img_tgt is not None: th,tw,_=img_tgt.shape
        else: th,tw=h,w
        pix_pos, msk = self.get_homography_correspondence(h, w, th, tw, H)
        pix_pos0, pix_pos1 = self.sample_correspondence(img, pix_pos, msk)
        return pix_pos0, pix_pos1

    @staticmethod
    def get_image_region(img, th, tw):
        th, tw=int(th), int(tw)
        h, w, _ = img.shape
        if h > th:
            hbeg = np.random.randint(0, h - th)
            hend = hbeg + th
        else:
            hbeg, hend = 0, h

        if w > tw:
            wbeg = np.random.randint(0, w - tw)
            wend = wbeg + tw
        else:
            wbeg, wend = 0, w

        return img[hbeg:hend, wbeg:wend], wbeg, hbeg

    @staticmethod
    def resize_image_to_min(img, th, tw):
        oh,ow,_=img.shape
        if th<=oh and tw<=ow:
            return img, 1.0

        ratio=max((th+1)/oh, (tw+1)/ow)
        img=cv2.resize(img,(int(ratio*ow),int(ratio*oh)))
        return img, ratio

    @staticmethod
    def compute_random_begin(beg,end,maxl,tagl):
        assert(tagl<=maxl)
        if tagl>(beg-end):
            beg_max=min(beg,end-tagl)
            if beg_max<=0: return 0
            else:
                return np.random.randint(0,beg_max)
        elif tagl==(beg-end): return beg
        else:
            print(beg,end-tagl,end,tagl)
            return np.random.randint(beg,end-tagl)

    @staticmethod
    def compute_suitable_region(img0, img1, H, th, tw):

        h0,w0,_=img0.shape
        h1,w1,_=img1.shape

        c1=np.asarray([[0,0],[0,h1],[w1,h1],[w1,0]],np.float32)
        c10=perspective_transform(c1,np.linalg.inv(H))
        minx,miny=np.min(c10,0).astype(np.int32)
        maxx,maxy=np.max(c10,0).astype(np.int32)

        wbeg0=CorrespondenceDataset.compute_random_begin(max(minx,0),min(maxx,w0),w0,tw)
        hbeg0=CorrespondenceDataset.compute_random_begin(max(miny,0),min(maxy,h0),h0,th)

        c0=np.asarray([[wbeg0,hbeg0],[wbeg0,hbeg0+th],[wbeg0+tw,hbeg0+th],[wbeg0+tw,hbeg0]],np.float32)
        c01=perspective_transform(c0,H)
        minx,miny=np.min(c01,0).astype(np.int32)
        maxx,maxy=np.max(c01,0).astype(np.int32)
        wbeg1=CorrespondenceDataset.compute_random_begin(max(minx,0),min(maxx,w1),w1,tw)
        hbeg1=CorrespondenceDataset.compute_random_begin(max(miny,0),min(maxy,h1),h1,th)

        return img0[hbeg0:hbeg0+th,wbeg0:wbeg0+tw], img1[hbeg1:hbeg1+th,wbeg1:wbeg1+tw], wbeg0, hbeg0, wbeg1, hbeg1

    def generate_background_pth(self, cur_pth):
        bpth=self.background_pths[np.random.randint(0,len(self.background_pths))]
        while bpth==cur_pth: bpth=self.background_pths[np.random.randint(0,len(self.background_pths))]
        return bpth

    def generate_homography(self):
        if self.args['use_superpoint']:
            if self.args['superpoint_refined']=='homography':
                H=sample_homography(self.args['h'],self.args['w'])
            elif self.args['superpoint_refined']=='homography_v2':
                H=sample_homography_v2(self.args['h'],self.args['w'])
            elif self.args['superpoint_refined']=='homography_test':
                H=sample_homography_test(self.args['h'],self.args['w'])
            elif self.args['superpoint_refined']=='identity':
                H=sample_identity(self.args['h'],self.args['w'])
            elif self.args['superpoint_refined']=='strictly_identity':
                H=np.identity(3)
            elif self.args['superpoint_refined']=='4_rotate_3_scale':
                H=sample_4_rotate_3_scale(self.args['h'],self.args['w'])
            elif self.args['superpoint_refined']=='old_homography':
                H=compute_homography(self.args['h'],self.args['w'],self.args['perspective'],self.args['scaling'],self.args['rotation'],
                                     self.args['translation'],self.args['min_patch_ratio'],self.args['max_patch_ratio'],
                                     self.args['perspective_amplitude_x'],self.args['perspective_amplitude_y'],
                                     self.args['scaling_amplitude'],self.args['rotate_max_angle'],allow_artifacts=self.args['allow_artifacts'])
            else: raise NotImplementedError
        else:
            H, _ = compute_homography_discrete(self.args['h'], self.args['w'],
                                                          self.args['scaling'], self.args['rotation'],
                                                          self.args['translation'], self.args['perspective'],
                                                          self.args['scale_base_ratio'],
                                                          self.args['scale_factor_range'],
                                                          self.args['max_scale_disturb'],
                                                          self.args['max_angle'] / 180 * np.pi,
                                                          self.args['max_translation'],
                                                          self.args['perspective_x'],
                                                          self.args['perspective_y'])
        return H

    def augment(self, img):
        # ['jpeg','blur','noise','jitter','none']
        if len(self.args['augment_classes']) > self.args['augment_num']:
            augment_classes = np.random.choice(self.args['augment_classes'], self.args['augment_num'],
                                               False, p=self.args['augment_classes_weight'])
        elif 0 < len(self.args['augment_classes']) <= self.args['augment_num']:
            augment_classes = self.args["augment_classes"]
        else:
            return img

        for ac in augment_classes:
            img = self.name2func[ac](img)
        return img

    # def compute_scale_rotate_offset(self, H, pix_pos0):
    #     As=compute_similar_affine_batch(H,pix_pos0)
    #     scale=np.sqrt(np.linalg.det(As))
    #     Us,_,Vs=np.linalg.svd(As)
    #     R=Us@Vs
    #     rotate=np.arctan2(R[:,1,0],R[:,0,0])
    #     scale_offset=np.round(np.log(scale)/np.log(self.base_scale)).astype(np.int32)
    #     rotate_offset=np.round(rotate/self.base_rotate).astype(np.int32)
    #     return scale_offset, rotate_offset
