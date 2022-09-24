from PIL import Image
import os
import numpy as np

scratch_path = '/cluster/scratch/jehrat/ai4good/jakarta'
folder_name = 'data'
#scratch_path = '/home/cupcake/Studium/semester_11/ai4g/ai4good_map/data'
#folder_name = 'test_png'

count = 0

for image_name in os.listdir(f'{scratch_path}/{folder_name}'):

    threshold = (300**2) * 0.9
    
    image = Image.open(folder_name +'/' + image_name)
    nparray = np.asarray(image)
    pixels = nparray[:, :, 0][nparray[:, :, 0] > 0.].shape[0]
    
    #print(image_name, ": ", pixels)
    
    if pixels <= threshold:
        os.remove(f'{scratch_path}/{folder_name}/{image_name}')
        print(image_name)
        count += 1

# print(count)
