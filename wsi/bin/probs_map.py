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

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

from wsi.data.wsi_producer import GridWSIPatchDataset  # noqa
from wsi.model import MODELS  # noqa
from wsi.utils.timer import Timer
import torch.multiprocessing
import matplotlib.pyplot as plt
import openslide
import matplotlib.patches as patches
import csv


# threshold applied AFTER the specified level
level_to_mask_threshold = {
    4: 0.01,
    3: 0.01,
    2: 0.01,
    1: 0.01, 
    0: 0.015 
}

parser = argparse.ArgumentParser(description='Get the probability map of tumor'
                                 ' patch predictions given a WSI')
parser.add_argument('wsi_path', default=None, metavar='WSI_PATH', type=str,
                    help='Path to the input WSI file')
parser.add_argument('ckpt_path', default=None, metavar='CKPT_PATH', type=str,
                    help='Path to the saved ckpt file of a pytorch model')
parser.add_argument('cfg_path', default=None, metavar='CFG_PATH', type=str,
                    help='Path to the config file in json format related to'
                    ' the ckpt file')
# Now computed in the dataloader
# parser.add_argument('mask_path', default=None, metavar='MASK_PATH', type=str,
                    # help='Path to the tissue mask of the input WSI file')
parser.add_argument('debug_path', default=None, metavar='DEBUG_PATH', type=str,
    help='Path to the output debug image showing bounding rectangles')

parser.add_argument('probs_map_path', default=None, metavar='PROBS_MAP_PATH',
                    type=str, help='Path to the output probs_map numpy file')
parser.add_argument('--GPU', default='0', type=str, help='which GPU to use'
                    ', default 0')
parser.add_argument('--num_workers', default=5, type=int, help='number of '
                    'workers to use to make batch, default 5')
parser.add_argument('--eight_avg', default=0, type=int, help='if using average'
                    ' of the 8 direction predictions for each patch,'
                    ' default 0, which means disabled')

#things that could be arguments if I wasnt lazy
DEBUG = True
# SKIP_MODEL = False

def threshholdMask(prevMask, currentLevel):
    newMask = np.zeros(prevMask.shape)
    thresh = level_to_mask_threshold[currentLevel]
    newMask[prevMask > thresh] = 1.0
    return newMask

# disclaimer: Not actually recursive but hey it could be
def recursive_probs_map(args, cfg, model, init_level):
    currentLevel = init_level
    previousMask = None
    firstTime = True
    allDebugInfo = []
    while (currentLevel != 0):
        # This will be the same size as the current mask so in the next image the spacing of 
        # searches will be close together if the probabilties are dense. Could change this up 
        # TODO: Try nms and/or something to get cluster centers and then just use the mask in 
        # an area around that? Would also want cluster height/width...
        dataloader = None
        if (not firstTime):
            # newMask = np.zeros_like(previousMask)
            newMask = threshholdMask(previousMask, currentLevel)
            # newMask = torch.where(previousMask > 0.5, 1., 0.)
            dataloader = make_dataloader(
                args, cfg, level=currentLevel, mask=newMask,
                flip='NONE', rotate='NONE' )
        else: 
            dataloader = make_dataloader(
                args, cfg, level=currentLevel, mask=None,
                flip='NONE', rotate='NONE' )
        print("entering get_probs_map for currentLevel =", currentLevel)
        previousMask, debugInfo = get_probs_map(model, dataloader)
        allDebugInfo = allDebugInfo + debugInfo
        print("prevmask shape:", previousMask.shape)
        currentLevel = currentLevel - 1
        firstTime = False


    newMask = threshholdMask(previousMask, currentLevel)
    # ie the final one is zero since the while loop exited,
    # I'll think of a better code pattern sometime
    dataloader = make_dataloader(
                args, cfg, level=currentLevel, mask=newMask,
                flip='NONE', rotate='NONE')
    probs_map, debugInfo = get_probs_map(model, dataloader)

    #np.asarray([np.array([140.0,140.0]), np.array([30.0,30.0])])
    good_points = np.where(probs_map > 0.9)

    if(DEBUG):
        plotAreasSearched(
            dataloader.dataset._wsi_path,
            (dataloader.dataset.X_mask, dataloader.dataset.Y_mask) ,
            allDebugInfo,
            args.debug_path,
            good_points,
            dataloader.dataset._resolution
        )

    return probs_map

def get_probs_map(model, dataloader):
    probs_map = np.zeros(dataloader.dataset._mask.shape)
    num_batch = len(dataloader)
    # only use the prediction of the center patch within the grid
    idx_center = dataloader.dataset._grid_size // 2
    
    count = 0
    allDebugInfo = []
    time_now = time.time()
    
    # needed because of allDebugInfo causing some kind of memory space issue for cuda/pytorch
    if DEBUG:
        torch.multiprocessing.set_sharing_strategy('file_system')

    for (data, x_mask, y_mask, debugInfo) in dataloader:
        allDebugInfo.append(debugInfo)
        data = Variable(data.cuda(async=True), volatile=True)
        
        output = model(data)
        
        # because of torch.squeeze at the end of forward in resnet.py, if the
        # len of dim_0 (batch_size) of data is 1, then output removes this dim.
        # should be fixed in resnet.py by specifying torch.squeeze(dim=2) later
        
        if len(output.shape) == 1:
            probs = output[idx_center].sigmoid().cpu().data.numpy().flatten()
        else:
            probs = output[:,
                           idx_center].sigmoid().cpu().data.numpy().flatten()

        probs_map[x_mask, y_mask] = probs
        count += 1

        time_spent = time.time() - time_now
        time_now = time.time()
        logging.info(
            '{}, flip : {}, rotate : {}, batch : {}/{}, Run Time : {:.2f}'
            .format(
                time.strftime("%Y-%m-%d %H:%M:%S"), dataloader.dataset._flip,
                dataloader.dataset._rotate, count, num_batch, time_spent))
        # if count == 4:
        #     break    

    return probs_map, allDebugInfo

def plotAreasSearched(wsi_path, mask_shape, debugInfoList, debug_image_path, good_points, resolution):
    fig,ax = plt.subplots(1)
    downsampled_image = None
    
    slide = openslide.OpenSlide(wsi_path)
    downsampled_image = np.asarray( slide.get_thumbnail(size=mask_shape) )

    ax.imshow(downsampled_image)
    plt.axis('off')
    for infoNum, debugInfo in enumerate(debugInfoList):
        print("Rectangle ", infoNum)
        # Display the image
        #because of the 20 batches, there are 19 more down the y axis of each side that I chose to ignore
        # for x, y in zip(debugInfo["corner"][0], debugInfo["corner"][1] ):
        x, y = (debugInfo["corner"][0][0] , debugInfo["corner"][1][0] )
        heightAndWidth = int(debugInfo["hw"][0]) # / resolution
        plotColor = debugInfo["color"][0]
        # if(infoNum == 0):
            # print("x,y:", x, y) these are wrong rn because resolution 
            # print("h/w:", heightAndWidth)
        # todo add resolution here maybe
        rect = patches.Rectangle((x, y), heightAndWidth, heightAndWidth,linewidth=1,edgecolor=plotColor,facecolor='none')
        ax.add_patch(rect)

    for x, y in zip(good_points[0], good_points[1]):
        # print(point)
        ax.scatter(x, y, c="red", zorder=10)

    fig.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
            hspace = 0, wspace = 0)
    plt.margins(0,0)
    fig.gca().xaxis.set_major_locator(plt.NullLocator())
    fig.gca().yaxis.set_major_locator(plt.NullLocator())
    fig.savefig(debug_image_path, format='png', bbox_inches = 'tight',
            pad_inches = 0, dpi=136.6)
    plt.close()

def make_dataloader(args, cfg, level, mask=None, flip='NONE', rotate='NONE'):
    batch_size = cfg['batch_size'] * 2
    num_workers = args.num_workers

    dataloader = DataLoader(
        GridWSIPatchDataset(args.wsi_path,
                            image_size=cfg['image_size'],
                            patch_size=cfg['patch_size'],
                            crop_size=cfg['crop_size'], normalize=True,
                            flip=flip, rotate=rotate,
                            level=level, mask=mask),
        batch_size=batch_size, num_workers=num_workers, drop_last=False)

    return dataloader

def determineInitialLevel(wsi_path):
    slide = openslide.OpenSlide(wsi_path)
    maxLevel = slide.level_count - 1 # lowest possible resolution
    
    mask_level_to_level = {
        10: 4,
        9: 3,
        8: 2,
        7: 1,
        6: 0,
        5: -1 #shouldn't get here, maybe error?
    }

    init_level = mask_level_to_level[maxLevel] # yeah global variable whatever
    print("Maximum level used is", maxLevel, "so initial zoom is ", init_level)
    slide.close()

    return init_level

def run(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU
    logging.basicConfig(level=logging.INFO)

    with open(args.cfg_path) as f:
        cfg = json.load(f)

    if cfg['image_size'] % cfg['patch_size'] != 0:
            raise Exception('Image size / patch size != 0 : {} / {}'.
                            format(cfg['image_size'], cfg['patch_size']))

    patch_per_side = cfg['image_size'] // cfg['patch_size']
    grid_size = patch_per_side * patch_per_side

    # mask = np.load(args.mask_path)
    ckpt = torch.load(args.ckpt_path)
    model = MODELS[cfg['model']](num_nodes=grid_size, use_crf=cfg['use_crf'])
    model.load_state_dict(ckpt['state_dict'])
    model = model.cuda().eval()

    init_level = determineInitialLevel(args.wsi_path)

    timer = Timer()

    if not args.eight_avg:
        # dataloader = make_dataloader(
        #     args, cfg, flip='NONE', rotate='NONE', INIT_LEVEL)
        timer.start("get_probs_map ")
        # probs_map = get_probs_map(model, dataloader)
        probs_map = recursive_probs_map(args, cfg, model, init_level)
        runtime = timer.stop()

        with open(r'thresholds_and_times.csv', 'a') as csvfile:
            fieldnames = ['time','thresh0', 'thresh1', 'thresh2', 'thresh3', 'thresh4', 'input_args']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            # TODO add type of test? Normal or tumor?
            writer.writerow({
                'time':runtime,
                'thresh0':level_to_mask_threshold[0],
                'thresh1': level_to_mask_threshold[1],
                'thresh2': level_to_mask_threshold[2],
                'thresh3': level_to_mask_threshold[3],
                'thresh4': level_to_mask_threshold[4], 
                'input_args': ' '.join(sys.argv[1:] )
                })
    else:
        probs_map = np.zeros(mask.shape)

        dataloader = make_dataloader(
            args, cfg, flip='NONE', rotate='NONE')
        probs_map += get_probs_map(model, dataloader)

        dataloader = make_dataloader(
            args, cfg, flip='NONE', rotate='ROTATE_90')
        probs_map += get_probs_map(model, dataloader)

        dataloader = make_dataloader(
            args, cfg, flip='NONE', rotate='ROTATE_180')
        probs_map += get_probs_map(model, dataloader)

        dataloader = make_dataloader(
            args, cfg, flip='NONE', rotate='ROTATE_270')
        probs_map += get_probs_map(model, dataloader)

        dataloader = make_dataloader(
            args, cfg, flip='FLIP_LEFT_RIGHT', rotate='NONE')
        probs_map += get_probs_map(model, dataloader)

        dataloader = make_dataloader(
            args, cfg, flip='FLIP_LEFT_RIGHT', rotate='ROTATE_90')
        probs_map += get_probs_map(model, dataloader)

        dataloader = make_dataloader(
            args, cfg, flip='FLIP_LEFT_RIGHT', rotate='ROTATE_180')
        probs_map += get_probs_map(model, dataloader)

        dataloader = make_dataloader(
            args, cfg, flip='FLIP_LEFT_RIGHT', rotate='ROTATE_270')
        probs_map += get_probs_map(model, dataloader)

        probs_map /= 8

    np.save(args.probs_map_path, probs_map)


def main():
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
