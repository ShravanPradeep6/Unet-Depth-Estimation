import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.dice_score import multiclass_dice_coeff, dice_coeff

''' NO LONGER USING DICE SCORE AT ALL
@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # predict the mask
            mask_pred = net(image)

            if net.n_classes == 1:
                assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            else:
                assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes['
                # convert to one-hot format
                mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)

    net.train()
    return dice_score / max(num_val_batches, 1)
'''

'''DEFINITIONS TO BE USED FOR LOSSES'''
def gradient_x(img):
    return img[:, :, :, :-1] - img[:, :, :, 1:]

def gradient_y(img):
    return img[:, :, :-1, :] - img[:, :, 1:, :]

def gradient_loss(pred, target):
    pred_dx = gradient_x(pred)
    pred_dy = gradient_y(pred)

    target_dx = gradient_x(target)
    target_dy = gradient_y(target)

    valid = (target > 0).float()

    # gradient is valid only if both pixels involved are valid
    valid_dx = valid[:, :, :, :-1] * valid[:, :, :, 1:]
    valid_dy = valid[:, :, :-1, :] * valid[:, :, 1:, :]

    loss_x = (torch.abs(pred_dx - target_dx) * valid_dx).sum() / (valid_dx.sum() + 1e-8)
    loss_y = (torch.abs(pred_dy - target_dy) * valid_dy).sum() / (valid_dy.sum() + 1e-8)

    return loss_x + loss_y
def depth_loss(pred, true_depth):
    valid = (true_depth > 0).float()

    base_map = F.mse_loss(pred, true_depth, reduction='none')
    base = (base_map * valid).sum() / (valid.sum() + 1e-8)

    grad = gradient_loss(pred, true_depth)

    return base + 0.01 * grad

@torch.inference_mode()
def evaluate_depth(model, dataloader, device, amp, max_batches=20): 
    model.eval()
    total = 0.0
    n = 0
    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break
        images = batch['image'].to(device=device, dtype=torch.float32)
        true_depth = batch['depth'].to(device=device, dtype=torch.float32)

        with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
            pred = model(images)
            loss = depth_loss(pred, true_depth)

        total += loss.item()
        n += 1
    model.train()
    return total / max(n, 1)