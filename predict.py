import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
import SimpleITK as sitk

from unet import UNet
from utils import resize_and_crop
from torchvision import transforms


def predict_img(net,
                full_img,
                out_threshold=0.5,
                use_gpu=False):



    img = np.expand_dims(full_img, axis = 0)

    
    X = torch.from_numpy(img).unsqueeze(0)
    X = X.float()

    if use_gpu:
        X = X.cuda()

    with torch.no_grad():
        output = net(X)


        output_probs = F.sigmoid(output).squeeze(0)



    transforms.ToPILImage(output_probs)
    full_mask = output_probs.cpu().numpy()

    total_mask = np.zeros((7, 48, 48))
    total_mask[1:, :, :] = full_mask
    total_mask[0] = out_threshold
    final_mask = np.argmax(total_mask, axis = 0)

    return final_mask

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', default='MODEL.pth',
                        metavar='FILE',
                        help="Specify the file in which is stored the model"
                             " (default : 'MODEL.pth')")
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+',
                        help='filenames of input images', required=True)

    parser.add_argument('--output', '-o', metavar='INPUT', nargs='+',
                        help='filenames of ouput images')
    parser.add_argument('--cpu', '-c', action='store_true',
                        help="Do not use the cuda version of the net",
                        default=False)
    parser.add_argument('--no-save', '-n', action='store_false',
                        help="Do not save the output masks",
                        default=False)
    parser.add_argument('--mask-threshold', '-t', type=float,
                        help="Minimum probability value to consider a mask pixel white",
                        default=0.3)

    return parser.parse_args()

def get_output_filenames(args):
    in_files = args.input
    out_files = []

    if not args.output:
        for f in in_files:
            pathsplit = os.path.splitext(f)
            out_files.append("{}_OUT{}".format(pathsplit[0], pathsplit[1]))
    elif len(in_files) != len(args.output):
        print("Error : Input files and output files are not of the same length")
        raise SystemExit()
    else:
        out_files = args.output

    return out_files

if __name__ == "__main__":
    args = get_args()
    in_files = args.input
    out_files = get_output_filenames(args)

    net = UNet(n_channels=1, n_classes=6)

    print("Loading model {}".format(args.model))

    if not args.cpu:
        print("Using CUDA version of the net, prepare your GPU !")
        net.cuda()
        net.load_state_dict(torch.load(args.model))
    else:
        net.cpu()
        net.load_state_dict(torch.load(args.model, map_location='cpu'))
        print("Using CPU version of the net, this may be very slow")

    print("Model loaded !")

    for i, fn in enumerate(in_files):
        print("\nPredicting image {} ...".format(fn))

        img = sitk.ReadImage(fn)
        img = sitk.GetArrayFromImage(img)


        result = np.zeros((24, 50, 50))
        
        for j in range(24):

            s = img[j, 1:49, 1:49]
            mask = predict_img(net=net,
                               full_img=s,
                               out_threshold=args.mask_threshold,
                               use_gpu=not args.cpu)

            result[j, 1:49, 1:49] = np.rot90(mask, k = 2)


        if not args.no_save:
            out_fn = out_files[i]
            result = sitk.GetImageFromArray(result) 
            sitk.WriteImage(result, out_files[i])

            print("Mask saved to {}".format(out_files[i]))
