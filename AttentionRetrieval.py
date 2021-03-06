"""
Train and test the NetVLAD network with Attention module
"""

from __future__ import print_function

import json
import random
import shutil
import warnings
from datetime import datetime
from math import ceil
from os import makedirs
from os.path import join, exists

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import GenerateDecs
import TestScript
import TrainScript
import arguments
import loadCkpt
from DataSet import loadDataset
from UnifiedModel import Backbone as netavlad


def get_clusters(cluster_set):
    nDescriptors = 50000
    nPerImage = 100
    nIm = ceil(nDescriptors / nPerImage)

    sampler = SubsetRandomSampler(np.random.choice(len(cluster_set), nIm, replace=False))
    data_loader = DataLoader(dataset=cluster_set, num_workers=opt.threads, batch_size=opt.cacheBatchSize,
                             shuffle=False, pin_memory=True, sampler=sampler)

    if not exists(join(opt.dataPath, 'centroids')):
        makedirs(join(opt.dataPath, 'centroids'))

    initcache = join(opt.dataPath, 'centroids',
                     opt.arch + '_' + cluster_set.dataset + '_' + str(opt.num_clusters) + '_desc_cen.hdf5')
    with h5py.File(initcache, mode='w') as h5:
        with torch.no_grad():
            model.eval()
            print('====> Extracting Descriptors')
            dbFeat = h5.create_dataset("descriptors", [nDescriptors, encoder_dim], dtype=np.float32)

            for iteration, (input, indices) in enumerate(data_loader, 1):
                input = input.to(device)
                image_descriptors = model.encoder(input).view(input.size(0), encoder_dim, -1).permute(0, 2, 1)

                batchix = (iteration - 1) * opt.cacheBatchSize * nPerImage
                for ix in range(image_descriptors.size(0)):
                    # sample different location for each image in batch
                    sample = np.random.choice(image_descriptors.size(1), nPerImage, replace=False)
                    startix = batchix + ix * nPerImage
                    dbFeat[startix:startix + nPerImage, :] = image_descriptors[ix, sample, :].detach().cpu().numpy()

                if iteration % 50 == 0 or len(data_loader) <= 10:
                    print("==> Batch ({}/{})".format(iteration,
                                                     ceil(nIm / opt.cacheBatchSize)), flush=True)
                del input, image_descriptors

        print('====> Clustering..')
        niter = 100
        import faiss
        kmeans = faiss.Kmeans(encoder_dim, opt.num_clusters, niter=niter, verbose=False)
        kmeans.train(dbFeat[...])

        print('====> Storing centroids', kmeans.centroids.shape)
        h5.create_dataset('centroids', data=kmeans.centroids)
        print('====> Done!')


def save_checkpoint(state, is_best, filename='checkpoint_'):
    model_out_path = join(opt.savePath, filename+str(state['epoch'])+'.pt')
    torch.save(state, model_out_path)
    if is_best:
        shutil.copyfile(model_out_path, join(opt.savePath, 'model_best.pt'))


if __name__ == "__main__":
    # ignore warnings -- UserWarning: Loky-backed parallel loops cannot be called in a multiprocessing, setting n_jobs=1
    warnings.filterwarnings("ignore")

    # get arguments from the json file or the command
    opt = arguments.get_args()
    print(opt)
    rv = arguments.RunningVariables(opt)

    # designate the device (CUDA) to train
    if opt.nGPU == 0 or not torch.cuda.is_available():
        if opt.mode is 'train':
            raise Exception("No GPU found, program terminated")
        device = torch.device("cpu")
        random.seed(opt.seed)
        np.random.seed(opt.seed)
        rv.set_device(device)
    else:
        device = torch.device("cuda")
        random.seed(opt.seed)
        np.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        rv.set_device(device)

    print('===> Loading dataset(s)')
    dataset_tuple = loadDataset.loadDataSet(opt.mode.lower(), opt.split.lower(), opt.dataset.lower(),
                                            opt.threads, opt.cacheBatchSize, opt.margin)
    rv.set_dataset(*dataset_tuple)

    print('===> Building model')
    model, encoder_dim, hook_dim = netavlad.get_netavlad_model(opt=opt, train_set=dataset_tuple[0],
                                                               whole_test_set=dataset_tuple[3])
    isParallel = False
    if opt.nGPU > 1 and torch.cuda.device_count() > 1:
        print('Available GPU num = ', torch.cuda.device_count())

        model.encoder = nn.parallel.DataParallel(model.encoder)
        if opt.withAttention:
            model.attention = nn.parallel.DataParallel(model.attention)
        if opt.mode.lower() != 'cluster':
            model.pool = nn.parallel.DataParallel(model.pool)
        isParallel = True

    # Read the previous training results
    if opt.resume:
        model, start_epoch, best_metric = loadCkpt.load_ckpt(opt, device, model)
    else:
        model = model.to(device)
    rv.set_model(model, encoder_dim, hook_dim)

    # Define Optimizer and Loss Functions
    if opt.mode.lower() == 'train':
        if opt.optim.upper() == 'ADAM':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                   lr=opt.lr, weight_decay=opt.weightDecay)  # , betas=(0,0.9))

            rv.set_optimizer(optimizer)

        elif opt.optim.upper() == 'SGD':
            # set the learning rate of attention parameters as 10 times of other parameters
            if opt.withAttention:
                # attention_params = list(map(id, model.attention.parameters()))
                # print(list(model.attention.parameters()))
                attention_train_params = filter(lambda p: p.requires_grad, model.attention.parameters())
                base_params = filter(lambda p: p.requires_grad, model.parameters())
                base_train_params = list(set(base_params) ^ set(attention_train_params))
                optimizer = optim.SGD([
                    {'params': base_train_params, 'lr': opt.lr, 'momentum': opt.momentum,
                     'weight_decay': opt.weightDecay},
                    {'params': attention_train_params, 'lr': opt.lr*10, 'momentum': opt.momentum,
                     'weight_decay': opt.weightDecay}
                ])
            else:
                optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                      lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weightDecay)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=opt.lrStep, gamma=opt.lrGamma)
            rv.set_optimizer(optimizer)
        else:
            raise ValueError('Unknown optimizer: ' + opt.optim)

        # The L2 distances is used in PyTorch, instead of its square.
        criterion = nn.TripletMarginLoss(margin=opt.margin, p=2, reduction='sum').to(device)  # size_average=False
        rv.set_criterion(criterion)

    # Execute test/cluster/train
    if opt.mode.lower() == 'test':
        print('===> Running evaluation step')
        if opt.saveDecs:
            GenerateDecs.generate(rv, opt)
        else:
            epoch = 1
            recalls = TestScript.test(rv, None, opt, epoch, write_tboard=False)

    elif opt.mode.lower() == 'cluster':
        print('===> Calculating descriptors and clusters')
        get_clusters(dataset_tuple[1])

    elif opt.mode.lower() == 'train':
        print('===> Training model')
        writer = SummaryWriter(
            log_dir=join(opt.runsPath, datetime.now().strftime('%b%d_%H-%M-%S') + '_' + opt.arch + '_' + opt.pooling))

        # write checkpoints in logdir
        logdir = writer.file_writer.get_logdir()
        # opt.savePath = join(logdir, opt.savePath)
        if not opt.resume:
            makedirs(opt.savePath)

        with open(join(opt.savePath, 'flags.json'), 'w') as f:
            f.write(json.dumps(
                {k: v for k, v in vars(opt).items()}
            ))
        print('===> Saving state to:', logdir)

        not_improved = 0
        best_score = best_metric if opt.resume else 0
        for epoch in range(opt.start_epoch + 1, opt.nEpochs + 1):
            if opt.optim.upper() == 'SGD':
                scheduler.step(epoch)
            TrainScript.train(rv, writer, opt, epoch)
            if (epoch % opt.evalEvery) == 0:
                recalls = TestScript.test(rv, writer, opt, epoch, write_tboard=True)
                is_best = recalls[5] > best_score
                if is_best:
                    not_improved = 0
                    best_score = recalls[5]
                else:
                    not_improved += 1

                save_checkpoint({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'recalls': recalls,
                    'best_score': best_score,
                    'optimizer': optimizer.state_dict(),
                    'parallel': isParallel,
                }, is_best)

                if opt.patience > 0 and not_improved > (opt.patience / opt.evalEvery):
                    print('Performance did not improve for', opt.patience, 'epochs. Stopping.')
                    break

        print("=> Best Recall@5: {:.4f}".format(best_score), flush=True)
        writer.close()
