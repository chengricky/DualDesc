from torch import nn

from DenseDesc.network.loss import triplet_loss_shem_as, triplet_loss_hem_as
from DenseDesc.network.models import Net4Conv3Pool128DimAvgRes
from DenseDesc.network.models import get_WResNet18


def build_extractor(cfg):
    name2extractor = {
        'Net4Conv3Pool128DimAvgRes': Net4Conv3Pool128DimAvgRes,
        'WResNet18': get_WResNet18,
    }
    return name2extractor[cfg['extractor_type']](cfg)


class FeatureNetworkWrapper(nn.Module):
    def __init__(self, cfg):
        super(FeatureNetworkWrapper, self, ).__init__()
        self.extractor = build_extractor(cfg)
        self.margin = cfg['loss_margin']
        self.hem_interval = cfg['hem_interval']
        self.loss_type = cfg['loss_type']

    def forward(self, img0, img1, pix_pos0, pix_pos1, hem_thresh):
        feats0 = self.extractor(img0)
        feats1 = self.extractor(img1)

        pool_num = img0.shape[-1] // feats0.shape[-1]
        pix_pos0 = (pix_pos0 + 0.5) / pool_num - 0.5
        pix_pos1 = (pix_pos1 + 0.5) / pool_num - 0.5

        hem_interval = max(int(self.hem_interval / pool_num + 0.5), 1)
        hem_thresh = max(hem_thresh / pool_num, pool_num * 2, 2)
        if self.loss_type == 'shem':
            triplet_loss, neg_rate, dist_pos, dist_neg = triplet_loss_shem_as(
                feats0, feats1, pix_pos0, pix_pos1, hem_interval, hem_thresh, self.margin)
        elif self.loss_type == 'hem':
            triplet_loss, neg_rate, dist_pos, dist_neg = triplet_loss_hem_as(
                feats0, feats1, pix_pos0, pix_pos1, hem_interval, hem_thresh, self.margin)
        else:
            raise NotImplementedError

        results = {
            'triplet_loss': triplet_loss,
            'neg_rate': neg_rate,
            'dist_pos': dist_pos,
            'dist_neg': dist_neg,
        }
        return results

