import torch


def l2_normalize(x, ratio=1.0, axis=1):
    '''

    :param feats: b,f,h,w
    :return:
    '''
    norm = torch.unsqueeze(torch.clamp(torch.norm(x, 2, axis), min=1e-6), axis)
    x = x/norm*ratio
    return x