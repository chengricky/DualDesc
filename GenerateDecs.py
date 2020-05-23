"""
Generate and save both global and local descriptors.
"""
import numpy as np
import torch
from torch.utils.data import DataLoader
from os import path
from UnifiedModel import Backbone as mdl
import cv2


def generate(rv, opt):
    numDb = rv.whole_test_set.dbStruct.numDb
    subdir = ['reference', 'query']
    test_data_loader = DataLoader(dataset=rv.whole_test_set, num_workers=opt.threads,
                                  batch_size=opt.cacheBatchSize, shuffle=False, pin_memory=True)
    size = None
    rv.model.eval()
    timeRGB=0
    timeIR=0
    timeN=0
    with torch.no_grad():
        print('====> Generating Descriptors')
        for iteration, (rgb, ir, indices) in enumerate(test_data_loader, 1):
            # GLOBAL Decs
            rgb = rgb.to(rv.device)
            # get Torch Script
            # traced_script_module = torch.jit.trace(rv.model.encoder, rgb)
            # traced_script_module.save('backbone.pt')
            import time
            since = time.time()
            image_encoding = rv.model.encoder(rgb)
            if opt.withAttention:
                att = rv.model.attention(image_encoding)
                vlad_encoding = rv.model.pool(image_encoding, att)
            else:
                vlad_encoding = rv.model.pool(image_encoding)
            time_elapsed = time.time() - since
            timeRGB += time_elapsed / rgb.shape[0]
            print('Extracting time per RGB (ms)', 1000 * time_elapsed/rgb.shape[0])

            if iteration % 50 == 0 or len(test_data_loader) <= 10:
                print("==> Batch-RGB ({}/{})".format(iteration, len(test_data_loader)), flush=True)

            for i in range(vlad_encoding.size()[0]):
                idx = int(indices[i]) % numDb
                savepth = path.join(opt.saveDecsPath+subdir[int(indices[i]) // numDb], str(idx).zfill(6)+'.rgb.npy')
                np.save(savepth, vlad_encoding[i, :].detach().cpu().numpy())
                cv2.imwrite(savepth+'att.jpg', att[i, :].permute(1, 2, 0).detach().cpu().numpy())

            mdl.hook_features.clear()
            del image_encoding, vlad_encoding

            # LOCAL Decs
            ir = ir.to(rv.device)
            # ir = rgb
            since = time.time()
            _t = rv.model.encoder(ir)
            local_feat = mdl.hook_features[-1]
            size = local_feat.shape[1:]
            time_elapsed = time.time() - since
            timeIR += time_elapsed/ir.shape[0]
            print('Extracting time per IR (ms)', 1000 * time_elapsed/ir.shape[0])
            timeN+=1

            if iteration % 50 == 0 or len(test_data_loader) <= 10:
                print("==> Batch-IR ({}/{})".format(iteration, len(test_data_loader)), flush=True)

            if len(local_feat.shape) == 3:
                local_feat = local_feat[np.newaxis, :]
            for j in range(local_feat.shape[0]):
                idx = int(indices[j]) % numDb
                savepth = path.join(opt.saveDecsPath+subdir[int(indices[j]) // numDb], str(idx).zfill(6)+'.ir.npy')

                np.save(savepth, local_feat[j, :, :, :])
            mdl.hook_features.clear()
            del ir, local_feat, _t

    print("Time RGB (ms): ", timeRGB/timeN*1000)
    print("Time IR (ms): ", timeIR / timeN*1000)

    with open(path.join(opt.saveDecsPath, "paras.txt"), 'w') as fw:
        for ele in size:
            fw.write(str(ele)+'\n')

    del test_data_loader




