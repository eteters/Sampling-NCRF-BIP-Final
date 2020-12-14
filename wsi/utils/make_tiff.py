import openslide
from PIL import Image

import numpy as np
import json
import libtiff
import sys
sys.setrecursionlimit(200000)








# Work in progress
filename = "Tumor_001.json"
real_image_name = "tumor_001.tif"
new_image_name = "tumor_mask_001.tiff"
level = 5


with open(filename, 'r') as f:
    tumor_dict = json.load(f)
    print("loaded json")
    slide = openslide.OpenSlide(real_image_name)
    print("opened slide")

    mask = np.zeros(slide.level_dimensions[level])
    print(slide.level_dimensions[level])
    print("closing slide")
    scaley  = slide.level_dimensions[0][0]/slide.level_dimensions[level][0]
    scalex  = slide.level_dimensions[0][1]/slide.level_dimensions[level][1]
    slide.close()

    tumors = tumor_dict["positive"]
    for tumor in tumors:
        vertices = tumor["vertices"]
        x_u = 0
        y_u = 0
        i=0
        for vertex in vertices:
            x, y = vertex[0], vertex[1]
            x_u +=int(x/(scalex))
            y_u +=int(y/(scaley))
            print(int(x/(scalex)),int(y/(scaley)))
            mask[int(x/(scalex)),int(y/(scaley))] = 1
            mask[int(x/(scalex))+1,int(y/(scaley))+1] = 1
            mask[int(x/(scalex))+1,int(y/(scaley))-1] = 1
            mask[int(x/(scalex))-1,int(y/(scaley))+1] = 1
            mask[int(x/(scalex))-1,int(y/(scaley))-1] = 1
            mask[int(x/(scalex))  ,int(y/(scaley))+1] = 1
            mask[int(x/(scalex))  ,int(y/(scaley))-1] = 1
            mask[int(x/(scalex))+1,int(y/(scaley))  ] = 1
            mask[int(x/(scalex))-1,int(y/(scaley))  ] = 1
            mask[int(x/(scalex)),int(y/(scaley))] = 1
            mask[int(x/(scalex))+2,int(y/(scaley))+2] = 1
            mask[int(x/(scalex))+2,int(y/(scaley))-2] = 1
            mask[int(x/(scalex))-2,int(y/(scaley))+2] = 1
            mask[int(x/(scalex))-2,int(y/(scaley))-2] = 1
            mask[int(x/(scalex))  ,int(y/(scaley))+2] = 1
            mask[int(x/(scalex))  ,int(y/(scaley))-2] = 1
            mask[int(x/(scalex))+2,int(y/(scaley))  ] = 1
            mask[int(x/(scalex))-2,int(y/(scaley))  ] = 1

            i+=1
        def fill(x,y,n):
            print(x, y)
            mask[x,y] = 1
            if(n > 30000):
                return
            if mask[x,y+1]==0:
                fill(x,y+1,n+1)
            if mask[x,y-1]==0:
                fill(x,y-1,n+1)
            if mask[x+1,y]==0:
                fill(x+1,y,n+1)
            if mask[x-1,y]==0:
                fill(x-1,y,n+1)


            return
        fill(int(x_u/i), int(y_u/i),1)

    print("creating image from mask")
    image = Image.fromarray(mask)
    print("saving image")
    image.save(new_image_name, "TIFF")
    #imsave(new_image_name, mask)
    
