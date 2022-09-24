import os
import numpy as np
dirs = ['satellite/usa', 'masks/streets', 'masks/buildings']

list_dirs = [np.array(os.listdir(dir)) for dir in dirs]

intersect = list_dirs[1]
intersect = np.intersect1d(intersect, list_dirs[0])
intersect = np.intersect1d(intersect, list_dirs[1])
intersect = np.intersect1d(intersect, list_dirs[2])


for list_dir, dir in zip(list_dirs, dirs):
    diffs = np.setdiff1d(list_dir, intersect)

    for diff in diffs:
        try:
            print(diff)
            os.remove(f'{dir}/{diff}')
        except:
            pass
