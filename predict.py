import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from utils.data_loading import BasicDataset
from unet import UNet
from utils.utils import plot_img_and_mask

def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor, is_depth=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu()
        output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear', align_corners=False)
        '''CHANGE FROM THRESHOLDING TO KEEPING THE PROBABILITIES (IN BOTH) 
        
        if net.n_classes > 1:
            #mask = output.argmax(dim=1) OG CODE FROM GITHUB
            mask = torch.softmax(output, dim=1)[0, 1]  # [1,2,H,W] # probability of class 1 (continuous 0..1)
        else: #MODEL WAS TRAINED WITH 2 CLASSES NOT 1 ..
            #mask = torch.sigmoid(output) > out_threshold
            #mask = torch.sigmoid(output)[0, 0] OUR CODE but we want logits instead
            mask = output[0,0]
    CHANGE TO KEEP THE OUTPUT AS DECIMALS'''
    
        #depth = torch.sigmoid(output)[0, 0]  # [H,W] in [0,1] not segmentation anymore

        # in predict_img, return both
        logits = output[0,0]
        depth  = torch.sigmoid(logits)
        
    #return mask[0].long().squeeze().numpy()
    #return depth.numpy().astype('float32') #return depth
    return logits.numpy().astype('float32'), depth.numpy().astype('float32')


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images', required=True)
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='Filenames of output images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=1, help='Number of classes')
    
    return parser.parse_args()


def get_output_filenames(args):
    def _generate_name(fn):
        return f'{os.path.splitext(fn)[0]}_OUT.png'

    return args.output or list(map(_generate_name, args.input))

'''THIS FUNCTION IS COOKED FOR DEPTH PERCEPTION, SPECIFICALLY LOOKS FOR INTS (SEGMENTATION)'''
def mask_to_image(mask: np.ndarray, mask_values):
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    for i, v in enumerate(mask_values):
        out[mask == i] = v

    return Image.fromarray(out)


if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    in_files = args.input
    out_files = get_output_filenames(args)

    net = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    state_dict = torch.load(args.model, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict)

    logging.info('Model loaded!')

    for i, filename in enumerate(in_files):
        logging.info(f'Predicting image {filename} ...')
        img = Image.open(filename)
        
        '''
        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device)
        '''
        logits, depth = predict_img(net=net,
                            full_img=img,
                            scale_factor=args.scale,
                            out_threshold=args.mask_threshold,
                            device=device)

        print("LOGITS min/max/mean:", logits.min(), logits.max(), logits.mean())
        print("DEPTH  min/max/mean:", depth.min(), depth.max(), depth.mean())

        ''' Change whole thing, junk mixed in with normal stuff .. 
        if not args.no_save:
            out_filename = out_files[i]
            CAN'T USE mask_to_image ANYMORE
            #result = mask_to_image(mask, mask_values)
            print("min/max/mean:", mask.min(), mask.max(), mask.mean()) #testing lines
            print("p1/p50/p99:", np.percentile(mask, [1, 50, 99])) #testing lines
            ''''''
            
            m = np.squeeze(mask)
            m = (m - m.min()) / (m.max() - m.min() + 1e-8)
            Image.fromarray((m * 255).astype(np.uint8)).save(out_filename)
            
            m = np.squeeze(mask)                      # already [0,1] from sigmoid
            m8 = np.clip(m * 255.0, 0, 255).astype(np.uint8)
            Image.fromarray(m8).save(out_filename)
             THIS ASSUMES SIGMOID HAS ALREADY BEEN APPLIED TO THE IMAGE
            mask = np.squeeze(mask)
            result = Image.fromarray((mask * 255).astype(np.uint8))
            result.save(out_filename)
            
            logging.info(f'Mask saved to {out_filename}')
            '''

        if not args.no_save:
            out_filename = out_files[i]

            # Debug stats on what you're saving
            print("DEPTH min/max/mean:", depth.min(), depth.max(), depth.mean())
            print("DEPTH p1/p50/p99:", np.percentile(depth, [1, 50, 99]))

            # Save sigmoid output directly (no per-image normalization)
            m = np.squeeze(depth)  # [H,W] in [0,1]
            m8 = np.clip(m * 255.0, 0, 255).astype(np.uint8)
            Image.fromarray(m8).save(out_filename)

            logging.info(f'Depth saved to {out_filename}')

        '''
        if args.viz:
            logging.info(f'Visualizing results for image {filename}, close to continue...')
            plot_img_and_mask(img, mask)
        '''
        if args.viz:
            Image.fromarray(m8).show(title="pred depth")
