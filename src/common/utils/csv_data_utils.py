import csv
from datetime import datetime
import os
import pandas as pd
import numpy as np
import logging
import sys

def get_time_df(n_nodes: int, base_path, suffix='') -> pd.DataFrame:
    """Reads the csv file and returns a DataFrame with travel times."""
    cost_file_name = f'{base_path}/cabdata/cab{n_nodes}_c{suffix}.csv'
    try:
        return pd.read_csv(cost_file_name, dtype={'fromnode': 'int16', 'tonode': 'int16', 'c': 'float32'})
    except FileNotFoundError:
        logging.error(f'file {cost_file_name} not found')
        sys.exit(0)
    

def get_demand_df(n_nodes: int, base_path, suffix='') -> pd.DataFrame:
    """Reads the csv file and returns a DataFrame with demands."""
    weight_file_name = f'{base_path}/cabdata/cab{n_nodes}_w{suffix}.csv'
    try:
        return pd.read_csv(weight_file_name, dtype={'fromnode': 'int16', 'tonode': 'int16', 'w': 'float32'})
    except FileNotFoundError:
        logging.error(f'file {weight_file_name} not found')
        sys.exit(0)

def get_time_matrix(df: pd.DataFrame, n: int) -> np.ndarray:
    """Generates a time matrix using a DataFrame."""
    df = df.pivot(index='fromnode', columns='tonode', values='c')
    return df.to_numpy()

def get_demand_matrix(df: pd.DataFrame, n: int) -> np.ndarray:
    """Generates a demand matrix using a DataFrame."""
    df = df.pivot(index='fromnode', columns='tonode', values='w')
    return df.to_numpy()

def get_time_and_demand_matrices(n_nodes: int, base_path, suffix=''):
    """Returns the time and demand matrices for given number of nodes."""
    if suffix is None: suffix = ''
    return (get_time_matrix(get_time_df(n_nodes, base_path, suffix), n_nodes), get_demand_matrix(get_demand_df(n_nodes, base_path, suffix), n_nodes))


RESULT_FILE_NAME = 'labeling_setting_results.csv'
HEADERS = [
    'execution_date','n','p','alpha','r','v','time','labeling_paths', 'total_columns', 'UB1', 'UB2'
]

def write_process_summary(general_params, experiment_time, labeling_total_paths, total_columns, heuristic_UB1, labeling_UB2,  result_file_name:str = RESULT_FILE_NAME, result_file_headers:list=HEADERS ):
    file_exists = os.path.isfile(result_file_name)
    with open(result_file_name, 'a', newline='') as res_file:
        writer = csv.writer(res_file, delimiter=',')
        if not file_exists:
            writer.writerow(result_file_headers)
        writer.writerow([
            datetime.now(),
            general_params.n,
            general_params.p,
            general_params.alpha,
            general_params.r,
            general_params.v,
            experiment_time,
            labeling_total_paths, 
            total_columns,
            heuristic_UB1, 
            labeling_UB2
        ])
        
def write_results_from_paths(general_params, results: list, reported_time, grasp_best_profit=None,  suffix='', path=''):
    path=path or '../paths'
    os.makedirs(path, exist_ok=True)
    file_name = f'{path}/data_{general_params.n}_{general_params.p}_{general_params.alpha}_{general_params.perc}_0.1_{general_params.r}_4{suffix}.csv'
    print(f'writing results to {file_name}')
    all_edges_no_diag = [(i, j) for i in range(general_params.n) for j in range(general_params.n) if i != j]
    with open(file_name, 'w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        # writer.writerow([round(finish-start,4)])
        grasp_best_profit and writer.writerow([f'heuristic_best_profit: {grasp_best_profit}'])
        writer.writerow([reported_time])
        writer.writerow(all_edges_no_diag)
        writer.writerows(results)
        
def write_dominated_counter_file(general_params, data, file_path=''):
    file_path=file_path or '../paths'
    os.makedirs(file_path, exist_ok=True)
    now = datetime.now()
    string = now.strftime('%Y%m%d%H%M%S')
    file_name = f'{file_path}/dominated_paths_{general_params.n}_{general_params.p}_{general_params.alpha}_{general_params.perc}_0.1_{general_params.r}_{string}.csv'
    print(f'printing statistics to {file_name}')
    with open(file_name, 'w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(data)

def write_labeling_iteration_info(general_params, iteration, UB, paths_found, iterations_info, file_path):
    file_path=file_path or '../iterations'
    os.makedirs(file_path, exist_ok=True)

    file_name = f'{file_path}/{general_params.n}-{general_params.p}-{general_params.alpha}-{general_params.v}-{general_params.r}-labeling_iteration.csv'
    with open(file_name, 'w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['iteration', 'UB', 'paths_found'])
        writer.writerows(iterations_info)
