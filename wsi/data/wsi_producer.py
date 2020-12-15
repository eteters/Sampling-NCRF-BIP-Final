import numpy as np
from torch.utils.data import Dataset
import openslide
import PIL
import matplotlib.patches as patches
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import image
from wsi.bin.tissue_mask import getTissueMask
import math

np.random.seed(0)

level_to_mask_level = {
    4: 10,
    3: 9,
    2: 8,
    1: 7,
    0: 6 
}

level_to_plot_color = {
    0:"orange",
    1:"g",
    2:"b",   
    3:"purple" # shouldn't get any more zoomed out than this right?
}

class GridWSIPatchDataset(Dataset):
    """
    Data producer that generate all the square grids, e.g. 3x3, of patches,
    from a WSI and its tissue mask, and their corresponding indices with
    respect to the tissue mask
    """
    def __init__(self, wsi_path, image_size=768, patch_size=256,
                 crop_size=224, normalize=True, flip='NONE', rotate='NONE',
                 skip=1, level=0, mask=None):
        """
        Initialize the data producer.

        Arguments:
            wsi_path: string, path to WSI file
            mask_path: string, path to mask file in numpy format
            image_size: int, size of the image before splitting into grid, e.g.
                768
            patch_size: int, size of the patch, e.g. 256
            crop_size: int, size of the final crop that is feed into a CNN,
                e.g. 224 for ResNet
            normalize: bool, if normalize the [0, 255] pixel values to [-1, 1],
                mostly False for debuging purpose
            flip: string, 'NONE' or 'FLIP_LEFT_RIGHT' indicating the flip type
            rotate: string, 'NONE' or 'ROTATE_90' or 'ROTATE_180' or
                'ROTATE_270', indicating the rotate type
        """
        self._wsi_path = wsi_path
        # I now compute this in this file with the imported func
        # self._mask_path = mask_path
        self._image_size = image_size
        self._patch_size = patch_size
        self._crop_size = crop_size
        self._normalize = normalize
        self._flip = flip
        self._rotate = rotate
        self.skip = skip
        self.X_mask = None, #I need these later
        self.Y_mask = None,
        self.level = level,
        self._mask = mask
        self._preprocess()

    def _preprocess(self):
        self.level = self.level[0] # this was a tuple for some reason
        print("level in is ", self.level)
        maskLevel = level_to_mask_level[self.level]
        if self._mask is None:
            self.isFirst = True # todo might need this 
            self._mask = getTissueMask(self._wsi_path, maskLevel) # again with the tuple!
       # TODO else block for bumping dimensions
        # else: 
        #     self.isFirst = False 
        #     #Gotta bump up the dimensions!
        #     x_mask, y_mask = self._mask.shape
        #     tempMask = np.zeros((x_mask * 2, y_mask * 2))
        #     temp_X, temp_Y = np.where(self._mask)  
        #     print("temp_X, temp_Y: ", temp_X, temp_Y) 
        #     for x, y in zip(temp_X , temp_Y ):
        #         tempMask[x * 2,y * 2] = 1 
        #     self._mask = tempMask
        
        self._slide = openslide.OpenSlide(self._wsi_path)
        # Tissue mask default is 6 so these two should be pretty different
        
        X_slide, Y_slide = self._slide.level_dimensions[self.level]
        X_mask, Y_mask = self._mask.shape
        self.X_mask, self.Y_mask = self._mask.shape
        if X_slide / X_mask != Y_slide / Y_mask:
            raise Exception('Slide/Mask dimension does not match ,'
                            ' X_slide / X_mask : {} / {},'
                            ' Y_slide / Y_mask : {} / {}'
                            .format(X_slide, X_mask, Y_slide, Y_mask))

        # so typically the "resolution" will be 2^6 I think
        # Why do they do a * 1.0 here? float?
        self._resolution = X_slide * 1.0 / X_mask
        print("resolution is ", self._resolution)
        if not np.log2(self._resolution).is_integer():
            raise Exception('Resolution (X_slide / X_mask) is not power of 2 :'
                            ' {}'.format(self._resolution))

        # OC comment not me: all the idces for tissue region from the tissue mask
        # from me: get the (x,y)'s from where the mask is true, but at a lower resolution so you won't go through everything in the image
        # (this could be slower if the mask was closer in level)
        self._X_idcs, self._Y_idcs = np.where(self._mask)
        self._idcs_num = len(self._X_idcs)

        if self._image_size % self._patch_size != 0:
            raise Exception('Image size / patch size != 0 : {} / {}'.
                            format(self._image_size, self._patch_size))
        self._patch_per_side = self._image_size // self._patch_size
        self._grid_size = self._patch_per_side * self._patch_per_side


    def __len__(self):
        # TODO comments for breaking into 4 
        # if self.isFirst:
        return int(self._idcs_num)    
        # else:
        #     return int(self._idcs_num) * 4

    def __getitem__(self, idx):
        x_mask = None
        y_mask = None

        # if self.isFirst:
        x_mask, y_mask = self._X_idcs[idx], self._Y_idcs[idx]
        # else:
            # x_mask, y_mask = self._X_idcs[math.floor(idx / 4)], self._Y_idcs[math.floor(idx / 4)]
        
        # print("x and y mask indices retreived: ", x_mask, y_mask)
        # if not self.isFirst:
        #     shifter = 4
        #     # top left
        #     if (idx % 4) == 0:
        #         x_mask = x_mask - shifter
        #         y_mask = y_mask - shifter
        #     #bot left 
        #     if (idx % 4) == 1:
        #         x_mask = x_mask - shifter
        #         y_mask = y_mask + shifter
        #     # top right 
        #     if (idx % 4) == 2:
        #         x_mask = x_mask + shifter
        #         y_mask = y_mask - shifter
        #     # bot right 
        #     if (idx % 4) == 3:
        #         x_mask = x_mask + shifter
        #         y_mask = y_mask + shifter

        # here it is important that we multiply by resolution...
        # this gets us the equivalent point in the big point system
        x_center = int((x_mask + 0.5) * self._resolution)
        y_center = int((y_mask + 0.5) * self._resolution)


        # TODO This comment block was for moving the quadrants to the right spot before I realize the resolution was a problem
        # x = 0
        # y = 0
        # save_index = idx
        
        #top left 
        # if (save_index % 4) == 0:
        #     print("Zero!")
        #     x = int(x_center - self._image_size)
        #     y = int(y_center - self._image_size)
        # #bot left
        # elif (save_index % 4) == 1:
        #     print("One!")
        #     x = int(x_center - self._image_size)
        #     y = int(y_center)
        # #top right
        # elif (save_index % 4) == 2:
        #     print("Two!")
        #     x = int(x_center)
        #     y = int(y_center - self._image_size)
        # #bot right
        # elif (save_index % 4) == 3:
        #     print("Three!")
        #     x = int(x_center)
        #     y = int(y_center)

        # This just gets the corner 
        x = int(x_center - self._image_size / 2)
        y = int(y_center - self._image_size / 2)

        # here is read_region(location, level, size) where we get from slide
        # level is another thing to change to sample from a zoomed-in version of 
        img = self._slide.read_region(
            (x, y), self.level, (self._image_size, self._image_size)).convert('RGB')

        if self._flip == 'FLIP_LEFT_RIGHT':
            img = img.transpose(PIL.Image.FLIP_LEFT_RIGHT)

        if self._rotate == 'ROTATE_90':
            img = img.transpose(PIL.Image.ROTATE_90)

        if self._rotate == 'ROTATE_180':
            img = img.transpose(PIL.Image.ROTATE_180)

        if self._rotate == 'ROTATE_270':
            img = img.transpose(PIL.Image.ROTATE_270)

        # PIL image:   H x W x C
        # torch image: C X H X W # I didn't make this comment but hey good to know 
        img = np.array(img, dtype=np.float32).transpose((2, 0, 1))

        if self._normalize:
            img = (img - 128.0)/128.0

        # flatten the square grid
        img_flat = np.zeros(
            (self._grid_size, 3, self._crop_size, self._crop_size),
            dtype=np.float32)

        idx = 0
        for x_idx in range(self._patch_per_side):
            for y_idx in range(self._patch_per_side):
                # center crop each patch
                x_start = int(
                    (x_idx + 0.5) * self._patch_size - self._crop_size / 2)
                x_end = x_start + self._crop_size
                y_start = int(
                    (y_idx + 0.5) * self._patch_size - self._crop_size / 2)
                y_end = y_start + self._crop_size
                img_flat[idx] = img[:, x_start:x_end, y_start:y_end]

                idx += 1

        # Resolution variable helps me draw the right size back onto the mask resolution
        # Note these come out as weird shaped tensors because DataLoader does
        # a bunch of parallel stuff 
        debugInfo = {
        "corner": (x / self._resolution ,y / self._resolution),
        "hw": self._image_size / self._resolution,
        "color": level_to_plot_color[self.level]
        }

        return (img_flat, x_mask, y_mask, debugInfo)
