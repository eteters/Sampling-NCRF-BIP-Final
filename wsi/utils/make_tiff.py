import OpenSlide
import imageio

# Work in progress 
filename = "Tumor_101.json"
image_name = "tumor_101.tiff"
with open(filename, 'r') as f:
    tumor_dict = json.load(f)

    slide = openslide.OpenSlide(image_name)
    mask = np.zeros(slide.level_dimensions[0])
    tumors = tumor_dict["positive"]
    for tumor in tumors:
    	vertices = tumor["vertices"]
    	for vertex in vertices:
    		x, y = vertex[0], vertex[1]
    		mask[x,y] = 1

    file = File("tiff.tiff")
    ImageIO