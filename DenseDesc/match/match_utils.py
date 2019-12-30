import cv2
from pyflann import FLANN

import numpy as np

from utils.extend_utils.extend_utils_fn import find_nearest_point_idx


def flann_match(feats_que, feats_ref, use_cuda=True):
    if use_cuda:
        idxs = find_nearest_point_idx(feats_ref, feats_que)
    else:
        flann = FLANN()
        idxs, dists = flann.nn(feats_ref, feats_que, 1, algorithm='linear', trees=4)

    return idxs

def get_correspondence_from_flow(kpt0, flow_01, image_shape):
    """
    kpt0: n,2 {xy style}
    flow_01: w,h,2
    image_shape : at least w,h
    """
    h, w = image_shape[0], image_shape[1]
    kpt01 = np.array(kpt0, dtype=np.float64)
    kpt0 = kpt0.copy().astype(np.int)
    kpt01[:, 0] = kpt0[:, 0] + flow_01[kpt0[:, 1], kpt0[:, 0]][:, 0] * w
    kpt01[:, 1] = kpt0[:, 1] + flow_01[kpt0[:, 1], kpt0[:, 0]][:, 1] * h
    return kpt01

def keep_valid_kps(kps, H, h, w, flow=None, feats=None):
    """     Default using H    """
    n, _ = kps.shape
    if flow is not None:
        shape = [h, w]
        warp_kps = get_correspondence_from_flow(kps, flow, shape)
    else:
        warp_kps = perspective_transform(kps, H)

    mask = (warp_kps[:, 0] >= 0) & (warp_kps[:, 0] < w) & \
           (warp_kps[:, 1] >= 0) & (warp_kps[:, 1] < h)
    if feats is not None:
        return kps[mask], feats[mask]
    else:
        return kps[mask]


def compute_match_pairs(feats0, kps0, feats1, kps1, H):
    # 0 to 1
    idxs = flann_match(feats0, feats1, True)
    pr01 = kps1[idxs]
    gt01 = perspective_transform(kps0, H)

    idxs = flann_match(feats1, feats0, True)
    pr10 = kps0[idxs]
    gt10 = perspective_transform(kps1, np.linalg.inv(H))

    return pr01, gt01, pr10, gt10


def compute_matching_score(feats0, kps0, feats1, kps1, H, thresh=16):
    n0, n1 = kps0.shape[0], kps1.shape[0]

    pr01, gt01, pr10, gt10 = compute_match_pairs(feats0, kps0, feats1, kps1, H)
    dist0 = np.linalg.norm(pr01 - gt01, 2, 1)
    dist1 = np.linalg.norm(pr10 - gt10, 2, 1)
    correct0 = dist0 < thresh
    correct1 = dist1 < thresh

    # 1 to 0
    return (np.sum(correct1) + np.sum(correct0)) / (n1 + n0), np.sum(correct1) + np.sum(correct0)


def detect_and_compute_sift(img):
    sift = cv2.xfeatures2d.SIFT_create()
    if len(img.shape)==3: img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    kp, des = sift.detectAndCompute(img,None)
    return kp, des

def detect_dog_keypoints(img):
    sift = cv2.xfeatures2d.SIFT_create()
    if len(img.shape)==3:
        img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    kps=sift.detect(img,None)
    kps_np=np.asarray([kp.pt for kp in kps])
    return np.round(kps_np).astype(np.int32)

def perspective_transform(pts, H):
    tpts = np.concatenate([pts, np.ones([pts.shape[0], 1])], 1) @ H.transpose()
    # todo: I don't know the reason why we should use abs, but experiments show this is correct
    tpts = tpts[:, :2] / np.abs(tpts[:, 2:])
    return tpts