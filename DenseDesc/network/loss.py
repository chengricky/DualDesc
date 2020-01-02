import torch
import torch.nn.functional as F

from DenseDesc.network.network_utils import l2_normalize

from DenseDesc.hard_mining.hard_example_mining_layer import hard_example_mining_layer, semi_hard_example_mining_layer, \
    knn_hard_example_mining_layer, knn_semi_hard_example_mining_layer, knn_cuda, knn_cuda_permute


def sample_hard_feature(feats, feats_pos, pix_pos, interval, thresh):
    '''

    :param feats:     [b,h,w,f] feature maps used to search negative example
    :param feats_pos: [b,n,f]   d_a
    :param pix_pos:   [b,n,2]   location of d_p on this feature map
    :param interval:
    :param thresh:
    :return:
    '''
    with torch.no_grad():
        pix_neg=hard_example_mining_layer(feats, feats_pos, pix_pos, interval, thresh, True).long() # b,n,3
    feats=feats.permute(0,2,3,1)
    feats_neg=feats[pix_neg[:,:,0],pix_neg[:,:,2],pix_neg[:,:,1]]   # b,n,f
    return feats_neg


def sample_semi_hard_feature(feats, dis_pos, feats_pos, pix_pos, interval, thresh, margin):
    '''

    :param feats:     [b,h,w,f] feature maps used to search negative example
    :param dis_pos:   [b,n]     |d_a-d_p|_2
    :param feats_pos: [b,n,f]   d_a
    :param pix_pos:   [b,n,2]   location of d_p on this feature map
    :param interval:
    :param thresh:
    :param margin:
    :return:
    '''
    with torch.no_grad():
        pix_neg=semi_hard_example_mining_layer(feats, dis_pos, feats_pos, pix_pos, interval, thresh, margin).long() # b,n,3
    feats=feats.permute(0,2,3,1)
    feats_neg=feats[pix_neg[:,:,0],pix_neg[:,:,2],pix_neg[:,:,1]]   # b,n,f
    return feats_neg


def normalize_coordinates(coords, h, w):
    h=h-1
    w=w-1
    coords=coords.clone().detach()
    coords[:, :, 0]-= w / 2
    coords[:, :, 1]-= h / 2
    coords[:, :, 0]/= w / 2
    coords[:, :, 1]/= h / 2
    return coords


def interpolate_feats(feats0, feats1, pix_pos0, pix_pos1):
    with torch.no_grad():
        pix_pos0_norm=normalize_coordinates(pix_pos0,feats0.shape[2],feats0.shape[3])
        pix_pos1_norm=normalize_coordinates(pix_pos1,feats1.shape[2],feats1.shape[3])
        pix_pos0_norm=torch.unsqueeze(pix_pos0_norm, 1) # b,1,n,2
        pix_pos1_norm=torch.unsqueeze(pix_pos1_norm, 1) # b,1,n,2

    feats_pos0 = F.grid_sample(feats0, pix_pos0_norm, 'bilinear', 'border')[:, :, 0, :] # b,f,n
    feats_pos1 = F.grid_sample(feats1, pix_pos1_norm, 'bilinear', 'border')[:, :, 0, :] # b,f,n

    feats_pos0=feats_pos0.permute(0,2,1)  # b,n,f
    feats_pos1=feats_pos1.permute(0,2,1)  # b,n,f

    return feats_pos0, feats_pos1


def clamp_loss_all(loss):
    """
    max(loss, 0) with hard-negative mining
    """
    loss = torch.clamp(loss, min=0.0) # b,n
    num = torch.sum(loss>1e-6).float()
    total_num=1
    for k in loss.shape: total_num*=k
    return torch.sum(loss)/(num+1), num/total_num


def triplet_loss_hem_as(feats0, feats1, pix_pos0, pix_pos1, interval, thresh, margin=0.0):
    '''

    :param feats0:      [b,f,h,w]
    :param feats1:      [b,f,h,w]
    :param pix_pos0:       [b,n,2]
    :param pix_pos1:       [b,n,2]
    :param interval:
    :param thresh:
    :param margin:
    :return:
    '''
    feats_pos0, feats_pos1 = interpolate_feats(feats0, feats1, pix_pos0, pix_pos1)
    feats_pos0 = l2_normalize(feats_pos0,axis=2)
    feats_pos1 = l2_normalize(feats_pos1,axis=2)

    dist_pos=torch.norm(feats_pos0 - feats_pos1, 2, 2)

    feats_neg0=sample_hard_feature(feats0,feats_pos0,pix_pos0,interval,thresh)
    feats_neg1=sample_hard_feature(feats1,feats_pos0,pix_pos1,interval,thresh)

    dist_neg00=torch.norm(feats_pos0-feats_neg0,2,2)
    dist_neg01=torch.norm(feats_pos0-feats_neg1,2,2)
    dist_neg10=torch.norm(feats_pos1-feats_neg0,2,2)
    dist_neg11=torch.norm(feats_pos1-feats_neg1,2,2)

    # triplet_loss: anchor swap
    loss00= dist_pos - dist_neg00 + margin
    loss01= dist_pos - dist_neg01 + margin
    loss10= dist_pos - dist_neg10 + margin
    loss11= dist_pos - dist_neg11 + margin

    loss00,rate00=clamp_loss_all(loss00)
    loss01,rate01=clamp_loss_all(loss01)
    loss10,rate10=clamp_loss_all(loss10)
    loss11,rate11=clamp_loss_all(loss11)

    return loss00+loss01+loss10+loss11, (rate00+rate01+rate10+rate11)/4.0, \
           torch.mean(dist_pos), torch.mean(dist_neg00+dist_neg01+dist_neg10+dist_neg11)/4.0


def triplet_loss_shem_as(feats0, feats1, pix_pos0, pix_pos1, interval, thresh, margin=0.0):
    '''

    :param feats0:      [b,f,h,w]
    :param feats1:      [b,f,h,w]
    :param pix_pos0:       [b,n,2]
    :param pix_pos1:       [b,n,2]
    :param interval:
    :param thresh:
    :param margin:
    :return:
    '''
    feats_pos0, feats_pos1 = interpolate_feats(feats0, feats1, pix_pos0, pix_pos1)
    feats_pos0 = l2_normalize(feats_pos0,axis=2)
    feats_pos1 = l2_normalize(feats_pos1,axis=2)

    dist_pos=torch.norm(feats_pos0 - feats_pos1, 2, 2)

    feats_neg0=sample_semi_hard_feature(feats0,dist_pos,feats_pos0,pix_pos0,interval,thresh,margin)
    feats_neg1=sample_semi_hard_feature(feats1,dist_pos,feats_pos0,pix_pos1,interval,thresh,margin)

    dist_neg00=torch.norm(feats_pos0-feats_neg0,2,2)
    dist_neg01=torch.norm(feats_pos0-feats_neg1,2,2)
    dist_neg10=torch.norm(feats_pos1-feats_neg0,2,2)
    dist_neg11=torch.norm(feats_pos1-feats_neg1,2,2)

    # triplet_loss: anchor swap
    loss00= dist_pos - dist_neg00 + margin
    loss01= dist_pos - dist_neg01 + margin
    loss10= dist_pos - dist_neg10 + margin
    loss11= dist_pos - dist_neg11 + margin

    loss00,rate00=clamp_loss_all(loss00)
    loss01,rate01=clamp_loss_all(loss01)
    loss10,rate10=clamp_loss_all(loss10)
    loss11,rate11=clamp_loss_all(loss11)

    return loss00+loss01+loss10+loss11, (rate00+rate01+rate10+rate11)/4.0, \
           torch.mean(dist_pos), torch.mean(dist_neg00+dist_neg01+dist_neg10+dist_neg11)/4.0
