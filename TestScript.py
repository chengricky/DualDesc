import numpy as np
import torch
from torch.utils.data import DataLoader
import faiss
import cv2
from os.path import join


def loadWPCA(reduced_dim):
    """
    Apply Dimension Reduction (PCA+Whitening)
    """
    wpca = torch.load('pca'+str(reduced_dim)+'.pth')
    from UnifiedModel.Backbone import L2Norm
    l2 = L2Norm()
    return torch.nn.Sequential(wpca, l2)


def test(rv, writer, opt, epoch=0, write_tboard=False):
    # wpca = loadWPCA(5120).to(rv.device)

    # TODO what if features dont fit in memory?
    test_data_loader = DataLoader(dataset=rv.whole_test_set, num_workers=opt.threads,
                                  batch_size=opt.cacheBatchSize, shuffle=False,
                                  pin_memory=True)

    rv.model.eval()
    with torch.no_grad():
        print('====> Extracting Features')
        pool_size = rv.encoder_dim
        if opt.pooling.lower() == 'netvlad':
            pool_size *= opt.num_clusters
        # print(pool_size)
        # pool_size = 5120
        dbFeat = np.empty((len(rv.whole_test_set), pool_size), dtype=np.float32)

        for iteration, (input, indices) in enumerate(test_data_loader, 1):
            input = input.to(rv.device)
            image_encoding = rv.model.encoder(input)
            del input
            if opt.withAttention:
                attention_score = rv.model.attention(image_encoding)
                image_encoding = rv.model.pool(image_encoding, attention_score)
                del attention_score
            else:
                image_encoding = rv.model.pool(image_encoding)

            # vlad_encoding = wpca(vlad_encoding.unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1)
            dbFeat[indices.detach().numpy(), :] = image_encoding.detach().cpu().numpy().astype('float32')
            if iteration % 50 == 0 or len(test_data_loader) <= 10:
                print("==> Batch ({}/{})".format(iteration, len(test_data_loader)), flush=True)

            del image_encoding
            # torch.cuda.empty_cache()

    del test_data_loader
    torch.cuda.empty_cache()

    # extracted for both db and query, now split in own sets
    qFeat = dbFeat[rv.whole_test_set.dbStruct.numDb:]
    dbFeat = dbFeat[:rv.whole_test_set.dbStruct.numDb]
    # np.delete(dbFeat, list(range(rv.whole_test_set.dbStruct.numDb, len(rv.whole_test_set))), axis=0)

    print('====> Building faiss index')
    # res = faiss.StandardGpuResources()  # use a single GPU
    # build a flat (CPU) index
    index_flat = faiss.IndexFlatL2(pool_size)
    # make it into a gpu index
    # gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)
    # add vectors to the index
    index_flat.add(dbFeat)

    # faiss_index = faiss.IndexFlatL2(pool_size)
    # faiss_index.add(dbFeat)

    print('====> Calculating recall @ N')
    n_values = [1, 5, 10, 20]
    # if opt.dataset.lower() == 'tokyo247':
    #     n_values = [10, 50, 100, 200]

    import time
    since=time.time()
    _, predictions = index_flat.search(qFeat, max(n_values))
    time_elapsed=time.time()-since
    print('serching time per query (ms)', 1000*time_elapsed/rv.whole_test_set.dbStruct.numQ)

    # predictionNew = []
    # if opt.dataset.lower() == 'tokyo247':
    #     for idx, pred in enumerate(predictions):
    #         keep = [True for pidx in pred if rv.whole_test_set.dbStruct.dbTimeStamp[pidx] != rv.whole_test_set.dbStruct.qTimeStamp[idx]]
    #         # or (not (eval_set.dbStruct.utmDb[pidx] == eval_set.dbStruct.utmQ[idx]).all())]
    #         pred_keep = [pred[idxiii] for idxiii, iii in enumerate(keep) if iii is True]
    #         predictionNew.append(pred_keep[:max(n_values) // 10])
    #     predictions = predictionNew
    #     n_values = [1, 5, 10, 20]
    # elif opt.dataset.lower() == 'pittsburgh':
    #     for idx, pred in enumerate(predictions):
    #         keep = [True for pidx in pred if not (eval_set.dbStruct.utmDb[pidx] == eval_set.dbStruct.utmQ[idx]).all()]
    #         pred_keep = [pred[idxiii] for idxiii, iii in enumerate(keep) if iii is True]
    #         predictionNew.append(pred_keep[:max(n_values)//10])
    #     predictions = predictionNew
    #     n_values = [1, 5, 10, 20]

    # for each query get those within threshold distance
    gt = rv.whole_test_set.get_positives()

    correct_at_n = np.zeros(len(n_values))
    gtValid = 0
    # TODO can we do this on the matrix in one go?
    for qIx, pred in enumerate(predictions):
        # print(pred)
        # print(gt[qIx])
        if gt[qIx].size:
            gtValid += 1
            for i, n in enumerate(n_values):
                # if n == 1:
                #     print(pred[0])
                #     db = cv2.imread(join(rv.whole_test_set.dbStruct.rootDir,
                #                          rv.whole_test_set.dbStruct.dbImage[pred[0]]))
                #     cv2.imshow("database", db)
                #     q = cv2.imread(join(rv.whole_test_set.dbStruct.rootDir,
                #                         rv.whole_test_set.dbStruct.qImage[qIx]))
                #     cv2.imshow("query", q)
                #     cv2.waitKey(1)
                #     cv2.waitKey()
                # if in top N then also in top NN, where NN > N
                if np.any(np.in1d(pred[:n], gt[qIx])):
                    correct_at_n[i:] += 1
                    break
    recall_at_n = correct_at_n / gtValid

    recalls = {}  # make dict for output
    for i, n in enumerate(n_values):
        recalls[n] = recall_at_n[i]
        print("====> Recall@{}: {:.4f}".format(n, recall_at_n[i]))
        if write_tboard:
            writer.add_scalar('Val/Recall@' + str(n), recall_at_n[i], epoch)

    return recalls
