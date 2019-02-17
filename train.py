import sys
import os
from optparse import OptionParser
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader

from eval import eval_net
from unet import UNet
from utils import  split_train_val, get_ids, resize_and_crop, channalize_mask, readimage

from tensorboardX import SummaryWriter

class BrainDataset(Dataset):

    def __init__(self, dir_img, dir_mask, data):
        self.dir_img = dir_img
        self.dir_mask = dir_mask
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        img = readimage(self.dir_img + self.data[idx] + '.nii')
        img = resize_and_crop(img)
        img = np.expand_dims(img, axis=0)

        mask = readimage(self.dir_mask + self.data[idx] + '_mask.nii')
        mask = resize_and_crop(mask)
        mask = channalize_mask(mask)
        mask = np.transpose(mask, axes=[2, 0, 1])

        return img.astype(np.float32), mask.astype(np.float32)

def train_net(net,
              epochs=5,
              batch_size=1,
              lr=0.1,
              val_percent=0.05,
              save_cp=True,
              gpu=False,
              img_scale=1):

    dir_img = 'data/train/'
    dir_mask = 'data/train_masks/'
    dir_checkpoint = 'checkpoints/'

    ids = get_ids(dir_img)
    iddataset = split_train_val(ids, val_percent)

    traindata = BrainDataset(dir_img, dir_mask, iddataset['train'])
    valdata = BrainDataset(dir_img, dir_mask, iddataset['val'])

    trainloader = DataLoader(traindata, batch_size = args.batchsize, shuffle = True, num_workers = 4)
    valloader = DataLoader(valdata, batch_size = 1, shuffle = False, num_workers = 2)

    print('''
    Starting training:
        Epochs: {}
        Batch size: {}
        Learning rate: {}
        Training size: {}
        Validation size: {}
        Checkpoints: {}
        CUDA: {}
    '''.format(epochs, batch_size, lr, len(iddataset['train']),
               len(iddataset['val']), str(save_cp), str(gpu)))

    N_train = len(iddataset['train'])

    optimizer = optim.SGD(net.parameters(),
                          lr=lr,
                          momentum=0.9,
                          weight_decay=0.0005)

    criterion = nn.BCELoss()
    
    epoch_tb = 0    # for tensorboard plotting
    
    for epoch in range(epochs):
        print('Starting epoch {}/{}.'.format(epoch + 1, epochs))

        if epoch % 4 == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] /= 2
                print('learning rate == ' + str(param_group['lr']))

        epoch_loss = 0

        for i, data in enumerate(trainloader):

            imgs = data[0]
            true_masks = data[1]

            if gpu:
                imgs = imgs.cuda()
                true_masks = true_masks.cuda()
            
            masks_pred = net(imgs)
            masks_probs = F.sigmoid(masks_pred)
            masks_probs_flat = masks_probs.view(-1)

            true_masks_flat = true_masks.view(-1)
            true_masks_flat = true_masks_flat.float()

            loss = criterion(masks_probs_flat, true_masks_flat)
            epoch_loss += loss.item()

            print('{0:.4f} --- loss: {1:.6f}'.format(i * batch_size / N_train, loss.item()))

            writer.add_scalar('loss - iter', loss.item(), epoch_tb)
            epoch_tb += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Epoch finished ! Loss: {}'.format(epoch_loss / (i + 1)))
        writer.add_scalar('epoch_loss - epoch', epoch_loss / (i + 1), epoch)

        if epoch % 7 == 0:
            val_dice = eval_net(net, valloader, gpu, threshold = 0.3)
            print('Validation Dice Coeff 0.3: {}'.format(val_dice))
            val_dice = eval_net(net, valloader, gpu, threshold = 0.4)
            print('Validation Dice Coeff 0.4: {}'.format(val_dice))
            val_dice = eval_net(net, valloader, gpu, threshold = 0.5)
            print('Validation Dice Coeff 0.5: {}'.format(val_dice))

        writer.add_scalar('val_dice_coeff - epoch', val_dice, epoch)

        if save_cp:
            torch.save(net.state_dict(),
                       dir_checkpoint + 'CP{}.pth'.format(epoch + 1))
            print('Checkpoint {} saved !'.format(epoch + 1))



def get_args():
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=5, type='int',
                      help='number of epochs')
    parser.add_option('-b', '--batch-size', dest='batchsize', default=1,
                      type='int', help='batch size')
    parser.add_option('-l', '--learning-rate', dest='lr', default=0.1,
                      type='float', help='learning rate')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu',
                      default=False, help='use cuda')
    parser.add_option('-c', '--load', dest='load',
                      default=False, help='load file model')
    parser.add_option('-s', '--scale', dest='scale', type='float',
                      default=0.5, help='downscaling factor of the images')

    (options, args) = parser.parse_args()
    return options

if __name__ == '__main__':
    args = get_args()

    net = UNet(n_channels=1, n_classes=6)

    writer = SummaryWriter()

    if args.load:
        net.load_state_dict(torch.load(args.load))
        print('Model loaded from {}'.format(args.load))

    if args.gpu:
        net.cuda()


    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  gpu=args.gpu,
                  img_scale=args.scale)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
