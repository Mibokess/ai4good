from get_data_from_name import get_data_from_name
import pandas as pd
import os

import warnings
warnings.filterwarnings("ignore")

cities = pd.read_csv('./CSV_Files/us_cities_states_counties.csv', sep='|')
cities.drop_duplicates(subset=['City'])

east_coast = [
    'Maine', 
    'New Hampshire', 
    'Massachusetts', 
    'Rhode Island', 
    'Connecticut',
    'New York', 
    'New Jersey', 
    'Delaware', 
    'Maryland', 
    'Virginia', 
    'North Carolina', 
    'South Carolina', 
    'Georgia' 
    'Florida'
]

east_coast_cities = cities[cities['State full'].isin(east_coast)]

names = east_coast_cities['City'] + ', ' + east_coast_cities['State full'] + ', ' + east_coast_cities['State short']
names = names.unique()

random_names = names.sample(100, random_state=36)

for name in random_names:
    os.makedirs('data', exist_ok=True)
    os.makedirs('data/east_coast', exist_ok=True)

    try:
        print(name)
        get_data_from_name(name, 'data/east_coast', get_satellite=False)
    except:
        print(f'Could not find: {name}')
        continue