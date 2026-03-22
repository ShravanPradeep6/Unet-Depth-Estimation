import argparse
import logging
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split, Subset
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter #tensorboard import
from datetime import datetime #tensorboard import

#import wandb
from evaluate import evaluate_depth #using evaluate_depth now
from unet import UNet
from utils.data_loading import BasicDataset, CarvanaDataset
from utils.dice_score import dice_loss

'''make conditional so can use in both colab and mac
dir_img = Path('./depth_dataset/images/')
dir_mask = Path('./depth_dataset/gt/')
dir_checkpoint = Path('./checkpoints/')   # (or wherever you want)
'''

'''
# Try local repo first, then Colab path
if Path('./depth_dataset').exists():
    base = Path('./depth_dataset')
elif Path('/content/depth_dataset').exists():
    base = Path('/content/depth_dataset')
else:
    raise FileNotFoundError("depth_dataset not found")

dir_img = base / 'images'
dir_mask = base / 'gt'
dir_checkpoint = Path('./checkpoints')
'''

PROJECT_ROOT = Path(__file__).resolve().parent

if (PROJECT_ROOT / 'depth_dataset').exists():
    base = PROJECT_ROOT / 'depth_dataset'
elif Path('/content/depth_dataset').exists():
    base = Path('/content/depth_dataset')
else:
    raise FileNotFoundError("depth_dataset not found")

dir_img = base / 'images'
dir_mask = base / 'gt'
dir_checkpoint = PROJECT_ROOT / 'checkpoints'

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

    base_map = F.smooth_l1_loss(pred, true_depth, reduction='none')
    base = (base_map * valid).sum() / (valid.sum() + 1e-8)

    grad = gradient_loss(pred, true_depth)

    return base + 0.1 * grad
'''END OF DEFINITIONS USED FOR LOSSES'''

def train_model(
        model,
        device,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
):
    ''' CARVANA DATASET MAKES NO SENSE FOR DEPTH
    # 1. Create dataset
    try: #not even using the carvana dataset so this shouldn't matter
        dataset = CarvanaDataset(dir_img, dir_mask, img_scale)
        
        CHANGED SO ONE EPOCH DOESN'T TAKE SO LONG ON MY COMPUTER, CAN CHANGE BACK
        
        dataset = Subset(dataset, range(1000))  # 1000 images only
    except (AssertionError, RuntimeError, IndexError):
        dataset = BasicDataset(dir_img, dir_mask, img_scale)
    '''
    dataset = BasicDataset(dir_img, dir_mask, img_scale)

    #dataset = Subset(dataset, range(1000))

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    pin_memory = (device.type == 'cuda') #pin_memory = true good for CUDA
    num_workers = 2 if device.type == 'cuda' else 0

    loader_args = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    '''
    ADDITION FOR TENSORBOARD
    '''
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"unet_lr{learning_rate}_bs{batch_size}_{timestamp}"
    writer = SummaryWriter(log_dir=f"runs/{run_name}")

    ''' WANDB not needed anymore
    # (Initialize logging)
    experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
 
    experiment.config.update(
        dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
             val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale, amp=amp)
    )
    '''

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.AdamW(model.parameters(),
                              lr=learning_rate, weight_decay=weight_decay, foreach=True)
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5) #we aren't using DICE score anymore, so use min of loss instead
    #grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    # No AMP for now (consistent behavior everywhere)
    grad_scaler = None
    '''CROSS ENTROPY LOSS MAKES NO SENSE TO USE ANYMORE'''
    #criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    #criterion = nn.SmoothL1Loss()
    '''loss section is all moved below now, makes more sense'''

    # 5. Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            accum = 8
            optimizer.zero_grad(set_to_none=True)
            
            for batch in train_loader:
                #images, true_masks = batch['image'], batch['mask'] CHANGED TO BELOW
                images, true_depth = batch['image'], batch['depth']
                #DEPTH DEBUGGING ADDITION
                #print(true_depth.min(), true_depth.max()) OR MAYBE NOT ..

                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'
                '''
                MAC CHANGE
                '''
                #images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last) CHANGING
                images = images.to(device=device, dtype=torch.float32) #CHANGED
                #true_masks = true_masks.to(device=device, dtype=torch.long) CHANGED TO BELOW
                true_depth = true_depth.to(device=device, dtype=torch.float32)


                '''WHOLE SECTION CHANGED, REFERENCES TO MASK CHANGED TO TRUE_DEPTH, USES SIGMOID
                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    pred = torch.sigmoid(model(images)) #uses sigmoid activation
                    if model.n_classes == 1:
                        loss = criterion(pred, true_depth)
                        #loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                    else: #shouldn't be triggered anyway .. wouldn't worry about it for now
                        loss = criterion(pred, true_depth)
                        loss += dice_loss(
                            F.softmax(pred, dim=1).float(),
                            F.one_hot(true_depth, model.n_classes).permute(0, 3, 1, 2).float(),
                            multiclass=True
                        )
                    '''
                #or actually don't even need that entier structure, classes should NEVER be more than 1
                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    '''
                    pred = model(images)

                    valid = (true_depth > 0)  # ignore zero pixels

                    loss_map = F.smooth_l1_loss(pred, true_depth, reduction='none')
                    loss = (loss_map * valid).sum() / (valid.sum() + 1e-8)
                    ''' #expanded upon in the function at top (depth_loss)

                    pred = model(images)
                    loss = depth_loss(pred, true_depth)

                ''' MAC CHANGE, MAKE EVERYTHING NO GRAD_SCALER
                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()
                '''
                #EVERYTHING BELOW IS WITH GRADSCALER
                #optimizer.zero_grad(set_to_none=True)
                loss = loss / accum
                loss.backward()
                ''' HUGE BLOCK TO LOG GRADIENTS ONTO TENSORBOARD AND VERIFY IF THEY'RE BIG OR SMALL'''
                # ---- GRADIENT DIAGNOSTICS (drop-in) ----
                import math

                total_norm_sq = 0.0
                max_abs_grad = 0.0
                nonfinite_count = 0
                zero_count = 0
                total_count = 0

                for p in model.parameters():
                    if p.grad is None:
                        continue
                    g = p.grad.detach()

                    # Count non-finite grads (NaN/Inf)
                    nonfinite_count += (~torch.isfinite(g)).sum().item()

                    # Global L2 norm and max abs grad
                    total_norm_sq += g.float().pow(2).sum().item()
                    max_abs_grad = max(max_abs_grad, g.float().abs().max().item())

                    # How many exact zeros (useful for spotting underflow / dead grads)
                    zero_count += (g == 0).sum().item()
                    total_count += g.numel()

                total_grad_norm = math.sqrt(total_norm_sq)
                zero_fraction = (zero_count / total_count) if total_count > 0 else 0.0

                writer.add_scalar("grad/total_norm", total_grad_norm, global_step)
                writer.add_scalar("grad/max_abs", max_abs_grad, global_step)
                writer.add_scalar("grad/zero_fraction", zero_fraction, global_step)
                writer.add_scalar("grad/nonfinite_count", nonfinite_count, global_step)

                if global_step % 50 == 0:
                    writer.flush()

                '''WANDB
                # If you want the same metrics in wandb too:
                experiment.log({
                    "grad/total_norm": total_grad_norm,
                    "grad/max_abs": max_abs_grad,
                    "grad/zero_fraction": zero_fraction,
                    "grad/nonfinite_count": nonfinite_count,
                    "step": global_step,
                    "epoch": epoch,
                })
                '''
                # ---- END GRADIENT DIAGNOSTICS ----
                '''END OF GRADIENT LOGGING'''

                if (global_step + 1) % accum == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                '''
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                '''
                '''
                TENSORBOARD ADDITION
                '''
                writer.add_scalar('Loss/train', loss.item(), global_step) #tensorboard training loss

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                division_step = (n_train // (5 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        '''
                        histograms = {}
                        for tag, value in model.named_parameters():
                            tag = tag.replace('/', '.')
                            if not (torch.isinf(value) | torch.isnan(value)).any():
                                histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                            if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                                histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())
                        '''
                        val_loss = evaluate_depth(model, val_loader, device, amp, depth_loss)

                        '''
                        TENSORBOARD ADDITION
                        '''
                        #writer.add_scalar('Dice/val', val_score, global_step)#tensorboard
                        writer.add_scalar('Loss/val', val_loss, global_step)
                        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], global_step) #tensorboard
                        scheduler.step(val_loss) #schedule LR based on the validation loss

                        # pred and true_depth are [B,1,H,W] in [0,1]
                        pred_vis = pred[0]                       # [1,H,W]
                        true_vis = true_depth[0]                 # [1,H,W]
                        err_vis  = (pred_vis - true_vis).abs()   # [1,H,W]

                        writer.add_image('Depth/true', true_vis, global_step)
                        writer.add_image('Depth/pred', pred_vis, global_step)
                        writer.add_image('Depth/abs_error', err_vis, global_step)

                        ''' CAN'T USE ANYMORE, change wandb stuff later
                        logging.info('Validation Dice score: {}'.format(val_score))
                        try:
                            experiment.log({
                                'learning rate': optimizer.param_groups[0]['lr'],
                                'validation Dice': val_score,
                                'images': wandb.Image(images[0].cpu()),
                                'masks': {
                                    'true': wandb.Image(true_masks[0].float().cpu()),
                                    'pred': wandb.Image(masks_pred.argmax(dim=1)[0].float().cpu()),
                                },
                                'step': global_step,
                                'epoch': epoch,
                                **histograms
                            })
                        except:
                            pass
                        '''


        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            '''
            CHANGED FOR SUBSET REASONS
            '''
            #state_dict['mask_values'] = dataset.mask_values 
            base_ds = dataset.dataset if isinstance(dataset, Subset) else dataset #changed
            '''DON'T HAVE MASK VALUES ANYMORE'''
            # state_dict['mask_values'] = base_ds.mask_values #changed
            torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f"Epoch {epoch} avg loss: {epoch_loss / len(train_loader)}")
            logging.info(f'Checkpoint {epoch} saved!')
    
    writer.close()        

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    '''Make default number of classes 1 for depth estimation'''
    parser.add_argument('--classes', '-c', type=int, default=1, help='Number of classes')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    #CHANGING
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('mps' if torch.backends.mps.is_available() else
    #                  'cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    model = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    '''
    CHANGE FOR MAC
    '''
    #model = model.to(memory_format=torch.channels_last) COMMENTING OUT FOR MAC

    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        state_dict.pop('mask_values', None)
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    model.to(device=device)
    try:
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()
        model.use_checkpointing()
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
        )
