from get_data_from_name import get_data_from_name
import pandas as pd
import numpy as np
import osmnx as ox

f = open('./CSV_Files/eastcoast_with_buildings.txt', 'r')

names = f.readlines()
names = [name.strip() for name in names]

#print(names)

for name in names:

    try:
        print(name)
        get_data_from_name(name, 'data/east_coast_w_builldings', get_satellite=True)
    except:
        print(f'Could not find: {name}')
        continue