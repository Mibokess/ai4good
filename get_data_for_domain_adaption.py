from get_data_from_name import get_data_from_name
import pandas as pd
import os

import warnings
warnings.filterwarnings("ignore")

cities = pd.read_csv('./CSV_Files/nevada.txt', header=None, sep='\n')

os.makedirs('data', exist_ok=True)
os.makedirs('data/nevada', exist_ok=True)

for name in cities[0]:
    try:
        print(name)
        get_data_from_name(name, 'data/nevada', get_satellite=True)

    except:
        print(f'Could not find: {name}')
        continue
