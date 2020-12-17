import sys
import os
import argparse
import logging
import json
import time


import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

from wsi.data.wsi_producer import GridWSIPatchDataset  # noqa
from wsi.model import MODELS  # noqa
from wsi.utils.timer import Timer
import torch.multiprocessing
import matplotlib.pyplot as plt
import openslide
import matplotlib.patches as patches
import csv
from wsi.bin.tissue_mask import getTissueMask



parser = argparse.ArgumentParser(description='Get the number of pixels after Otsu thresholding at level 7 and 6.')

parser.add_argument('wsi_path', default=None, metavar='WSI_PATH', type=str,
                    help='Path to the input WSI file')

def run(args):
	slide = openslide.OpenSlide(args.wsi_path)
	mask7 = getTissueMask(args.wsi_path, 7)
	mask6 = getTissueMask(args.wsi_path, 6)

	print("Number of pixels from level 6 are ", mask6[mask6 == 1].size, "That is out of ", mask6.shape[0] * mask6.shape[1], "total pixels in the mask")
	print("Number of pixels from level 7 are ", mask7[mask7 == 1].size, "That is out of ", mask7.shape[0] * mask7.shape[1], "total pixels in the mask")
	print("Base resolution had ", slide.level_dimensions[0][0] * slide.level_dimensions[0][1], "total pixels!")


def main():
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()