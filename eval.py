import torch
import torch.nn.functional as F
import numpy as np



def dechannalize(mask):
    
    new_mask = torch.zeros((48, 48))
    for i in range(0, 6):

        for j in range(0, 48):

            for k in range(0, 48):

                if mask[0, i, j, k]:

                    new_mask[j, k] = i + 1
    
    return new_mask


def eval_net(net, dataset, gpu=False, threshold=0.1):

    tot = 0
    inter = 0
    union = 0
    for i, data in enumerate(dataset):
        img = data[0]
        true_mask = data[1]


        if gpu:
            img = img.cuda()
            true_mask = true_mask.cuda()

        mask_pred = net(img)
        mask_pred = (F.sigmoid(mask_pred) > threshold).float()

        dc_mask_pred = dechannalize(mask_pred)
        dc_true_mask = dechannalize(true_mask)
# dice loss
        for j in range(2304):
            
            dc_mask_pred_flat = dc_mask_pred.view(-1)
            dc_true_mask_flat = dc_true_mask.view(-1)

            if dc_mask_pred_flat[j]:
                if dc_mask_pred_flat[j] == dc_true_mask_flat[j]:
                    inter += 1

        union += (dc_mask_pred > 0.1).sum() + (dc_true_mask > 0.1).sum()
    
    return float(2 * inter) / union.float()
