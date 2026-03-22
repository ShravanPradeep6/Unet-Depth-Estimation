import logging
import numpy as np
import torch
from PIL import Image
from functools import lru_cache
from functools import partial
from itertools import repeat
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm


def load_image(filename):
    ext = splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    else:
        return Image.open(filename)


def unique_mask_values(idx, mask_dir, mask_suffix):
    mask_file = list(mask_dir.glob(idx + mask_suffix + '.*'))[0]
    mask = np.asarray(load_image(mask_file))
    if mask.ndim == 2:
        return np.unique(mask)
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])
        return np.unique(mask, axis=0)
    else:
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, scale: float = 1.0, mask_suffix: str = ''):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if isfile(join(images_dir, file)) and not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        
        # keep only ids that have exactly one matching GT file
        filtered = []
        missing = 0
        multiple = 0

        '''
        for _id in self.ids:
            matches = list(self.mask_dir.glob(_id + self.mask_suffix + '.*'))
            if len(matches) == 1:
                filtered.append(_id)
            elif len(matches) == 0:
                missing += 1
            else:
                multiple += 1

        if missing or multiple:
            logging.warning(f"Filtered dataset: missing_gt={missing}, multiple_gt={multiple}")

        self.ids = filtered
        logging.info(f"Final paired examples: {len(self.ids)}")

        logging.info(f'Creating dataset with {len(self.ids)} examples')
        '''
        self.ids = [
            splitext(file)[0]
            for file in listdir(images_dir)
            if isfile(join(images_dir, file)) and not file.startswith('.')
        ]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

        # Build one-time lookup tables instead of globbing thousands of times
        self.image_lookup = {}
        for file in listdir(images_dir):
            if file.startswith('.'):
                continue
            full_path = join(images_dir, file)
            if isfile(full_path):
                self.image_lookup[splitext(file)[0]] = Path(full_path)

        self.mask_lookup = {}
        for file in listdir(mask_dir):
            if file.startswith('.'):
                continue
            full_path = join(mask_dir, file)
            if isfile(full_path):
                stem = splitext(file)[0]
                self.mask_lookup[stem] = Path(full_path)

        filtered = []
        missing = 0

        for _id in self.ids:
            key = _id + self.mask_suffix
            if key in self.mask_lookup:
                filtered.append(_id)
            else:
                missing += 1

        if missing:
            logging.warning(f"Filtered dataset: missing_gt={missing}")

        self.ids = filtered
        logging.info(f"Final paired examples: {len(self.ids)}")
        '''
        logging.info('Scanning mask files to determine unique values')
        with Pool() as p:
            unique = list(tqdm(
                p.imap(partial(unique_mask_values, mask_dir=self.mask_dir, mask_suffix=self.mask_suffix), self.ids),
                total=len(self.ids)
            ))

        self.mask_values = list(sorted(np.unique(np.concatenate(unique), axis=0).tolist()))
        logging.info(f'Unique mask values: {self.mask_values}')
        '''

    def __len__(self):
        return len(self.ids)

    '''CHANGE PREPROCESSING FOR OUR DEPTH USE (NO MASK and stuff)
    @staticmethod
    def preprocess(mask_values, pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img = np.asarray(pil_img)

        if is_mask:
            mask = np.zeros((newH, newW), dtype=np.int64)
            for i, v in enumerate(mask_values):
                if img.ndim == 2:
                    mask[img == v] = i
                else:
                    mask[(img == v).all(-1)] = i

            return mask

        else:
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))

            if (img > 1).any():
                img = img / 255.0

            return img
    '''

    '''
    OVERHAULED preprocess METHOD (getting image in right format, data type)
    '''
    @staticmethod
    def preprocess(pil_img, scale, is_depth: bool):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'

        if is_depth:
            # Force grayscale first (fixes '1' + 'P' modes)
            pil_img = pil_img.convert('L')
            pil_img = pil_img.resize((newW, newH), resample=Image.BILINEAR)

            arr = np.asarray(pil_img, dtype=np.uint8)  # 0..255
            # Debug (optional)
            # print("DEPTH mode:", pil_img.mode, "min/max:", arr.min(), arr.max(), "unique:", np.unique(arr).size)

            depth = arr.astype(np.float32) / 255.0     # [H,W] -> [0,1]
            return depth[None, ...]                    # [1,H,W]

        else:
            pil_img = pil_img.resize((newW, newH), resample=Image.BICUBIC)
            arr = np.asarray(pil_img)

            if arr.ndim == 2:
                arr = arr[np.newaxis, ...]
            else:
                arr = arr.transpose((2, 0, 1))

            if (arr > 1).any():
                arr = arr / 255.0

            return arr.astype(np.float32)

    def __getitem__(self, idx):
        name = self.ids[idx]
        #mask_file = list(self.mask_dir.glob(name + self.mask_suffix + '.*'))
        #img_file = list(self.images_dir.glob(name + '.*'))
        img_file = self.image_lookup.get(name, None)
        mask_file = self.mask_lookup.get(name + self.mask_suffix, None)

        '''
        CHANGE ASSERTS
        
        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        '''
        #if len(img_file) != 1:
            #raise RuntimeError(f'Image issue for ID {name}: {img_file}')

        '''
        if len(mask_file) != 1:
            # Soft skip: pick a new random index instead of crashing a whole run
            # (this can happen if there are a few broken pairs)
            new_idx = np.random.randint(0, len(self.ids))
            return self.__getitem__(new_idx)
        depth = load_image(mask_file[0])
        img = load_image(img_file[0])
        '''
        if img_file is None:
            raise RuntimeError(f'Image issue for ID {name}: not found')

        if mask_file is None:
            raise RuntimeError(f'Mask issue for ID {name}: not found')

        depth = load_image(mask_file)
        img = load_image(img_file)

        assert img.size == depth.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {depth.size}'


        #MODIFIED BELOW TO MAKE IMAGES SUITABLE FOR DEPTH (FLOAT DEPTH PIXELS, ETC.)
        img = self.preprocess(img, self.scale, is_depth=False)     # np float32 [3,H,W]
        depth = self.preprocess(depth, self.scale, is_depth=True)  # np float32 [1,H,W]

        return {
            'image': torch.from_numpy(img).contiguous(),
            'depth': torch.from_numpy(depth).contiguous()
        }

class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, mask_dir, scale=1):
        super().__init__(images_dir, mask_dir, scale, mask_suffix='_mask')
