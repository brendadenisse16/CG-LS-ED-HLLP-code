import numpy as np
import pandas as pd
from datetime import datetime
import sys

def read_cabdata(n, suffix=''):
    distance_file = f'cabdata/cab{n}_c{suffix}.csv'
    demand_file = f'cabdata/cab{n}_w{suffix}.csv'

    # Read the CSV files
    distance_df = pd.read_csv(distance_file)
    demand_df = pd.read_csv(demand_file)

    # Pivot the data to create matrices
    distance_matrix = distance_df.pivot(index='fromnode', columns='tonode', values='c').values
    demand_matrix = demand_df.pivot(index='fromnode', columns='tonode', values='w').values

    return distance_matrix, demand_matrix


def process_raw_path(str_raw_path):
    path = np.array(eval(str_raw_path))-1
    return path

def process_raw_path_labeling(str_raw_path):
    path = np.array(eval(str_raw_path))
    return path

def get_hubs(path, type):
    cases = {
        1: path,
        2: path[1:],
        3: path[:-1],
        4: path[1:-1]
    }
    return cases.get(type, None)

def ifnull(pos, val):
    if len(sys.argv) <= pos or (sys.argv[pos] == None):
        return val
    return sys.argv[pos]

def get_current_datetime():
    return datetime.now().strftime("%d/%m/%Y %H:%M:%S")

def removeDuplicates(lst):      
    return [t for t in (set(tuple(i) for i in lst))]


