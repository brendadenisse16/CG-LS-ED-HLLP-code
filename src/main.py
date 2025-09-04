import os
import time
from datetime import datetime
import pandas as pd

from common.structures.parameters import Parameters
from common.structures.general_info import GeneralInfo
import logging as lg
from common.utils.csv_data_utils import write_process_summary, write_results_from_paths
from common.utils.data_utils import ifnull


from logic.heuristic import get_heuristic_paths
from logic.labeling_setting import generate_paths
from logic.rmp import main_rmp

import threading
import time
import signal
import sys

DEFAULT_PARAMS = {
  'N':20,
  'P':7,
  'ALPHA':0.2,
  'V': 0.1,
  'R': 1.7,
  'MAX_NEIGHBOURS_EXTENSION':5,  # used in the column generation process. How many neighbours to consider on each extension. The MN closer to the extension point are used
  'MAX_EXECUTION_TIME':36000, # The maximun processing time allowed.  If reached, the program will write results and then exit.
  'MAX_LABELS_PER_COMODITY':100, #  Controls the maximum number of labes to be identifed per comodity in each column generation process.
  'IS_HYBRID':'True', # True or False. If false the column generation will use the heuristic approach only.  Heuristic + exact 
  'MAX_HEURISTIC_ITERATIONS':30, # 
  'EXACT_NEIGHBOURS_EXTENSION':999, # 
  'PROBLEM_TYPE':'CABDATA',
}

BASE_PATH=''

# GENERAL GLOBAL PARAMETERS
MAIN_START_TIME = 0
LABELING_ITERATION_COUNTER = 0

# PARAMETROS HEURISTICA INICIAL
MAX_HEURISTIC_ITER_PER_COMODITY = 5#100
MAX_PATHS_PER_COMODITY_PER_TYPE = 1 #10

# PARAMETROS DE LABELING (default)
MAX_NEIGHBOURS_EXTENSION = DEFAULT_PARAMS.get('MAX_NEIGHBOURS_EXTENSION') 
MAX_LABELS_PER_COMODITY = DEFAULT_PARAMS.get('MAX_LABELS_PER_COMODITY')


MAX_HEURISTIC_ITERATIONS = DEFAULT_PARAMS.get('MAX_HEURISTIC_ITERATIONS')
EXACT_NEIGHBOURS_EXTENSION = DEFAULT_PARAMS.get('EXACT_NEIGHBOURS_EXTENSION')

# EXECUTION TIME CONTROL (IN SECONDS, IF REACHED THE PROCESS WRITES WHATEVER RESULTS IT HAS)
# MAX_EXECUTION_TIME = 36000

# ENABLES THE HYBRID MODE 
# FALSE -> HERUSITIC
# TRUE  -> EXACT
IS_HYBRID= DEFAULT_PARAMS.get('IS_HYBRID')
PROBLEM_TYPE = DEFAULT_PARAMS.get('PROBLEM_TYPE')


def get_base_path():
  return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
  

def clean_concat_paths_df(paths_df1, paths_df2):
  res_df = pd.concat([paths_df1,paths_df2], ignore_index=True)
  res_df['path'] = res_df['path'].apply(tuple)
  res_df.drop_duplicates(subset=['path','type'], keep='first',ignore_index=True, inplace=True)
  res_df['path'] = res_df['path'].apply(list)
  return res_df

UB1=0
UB2=0
TOTAL_HEURISTIC_PATHS=0
columns=pd.DataFrame()

def main_labeling(general_params:Parameters, general_info:GeneralInfo, base_path):
  global columns
  global LABELING_ITERATION_COUNTER
  global MAX_NEIGHBOURS_EXTENSION
  
  main_start = time.perf_counter()  
  heuristic_res = get_heuristic_paths(general_params, general_info, MAX_HEURISTIC_ITER_PER_COMODITY,MAX_PATHS_PER_COMODITY_PER_TYPE)
  
  write_results_from_paths(general_params, heuristic_res.to_numpy(), round(time.perf_counter() - main_start,2), 0,  '_heuristic', f'{base_path}/paths/heuristic' )
  
  paths_found_len = len(heuristic_res)
  lg.info(f'Heuristics found {paths_found_len} paths!')
  
  d_v, UB1 = main_rmp(general_params, general_info,heuristic_res, LABELING_ITERATION_COUNTER)
  TOTAL_HEURISTIC_PATHS=len(heuristic_res)
  lg.info(f'[{LABELING_ITERATION_COUNTER}]UB1:{UB1}, paths found:{TOTAL_HEURISTIC_PATHS}')
  total_paths = 0
  columns = heuristic_res.copy()
  
  while(True):
    LABELING_ITERATION_COUNTER+=1
    
    paths=generate_paths(d_v, general_info, MAX_NEIGHBOURS_EXTENSION, MAX_LABELS_PER_COMODITY)
    columns = clean_concat_paths_df(columns, paths)
    #lg.info(f'##### IS_HYBRID: {IS_HYBRID}')
    if IS_HYBRID:
      # CONDICION DE SALIDA PRINCIPAL ***
      if paths.empty or abs((total_paths -len(columns))/len(columns))<=0.0002:
        if MAX_NEIGHBOURS_EXTENSION < EXACT_NEIGHBOURS_EXTENSION: # means we were running the heuristic
          lg.info(f'Heuristic solution reached, switching to exact mode ({EXACT_NEIGHBOURS_EXTENSION})')
          MAX_NEIGHBOURS_EXTENSION = EXACT_NEIGHBOURS_EXTENSION
        else:
          lg.info(f'problem solution')
          break #No hay nuevos caminos    
      total_paths = len(columns)
      
      # GENERATE DUAL VALUES USING RMP
      d_v, UB2 = main_rmp(general_params, general_info,columns,LABELING_ITERATION_COUNTER)
      
      lg.info(f'[{LABELING_ITERATION_COUNTER}]UB2:{UB2}, paths found:{len(paths)} total unique paths: {len(columns)}') 
      
      if (LABELING_ITERATION_COUNTER >= MAX_HEURISTIC_ITERATIONS and MAX_NEIGHBOURS_EXTENSION < EXACT_NEIGHBOURS_EXTENSION):
        lg.info(f'Max heuristic iterations reached ({MAX_HEURISTIC_ITERATIONS}), switching to exact mode ({EXACT_NEIGHBOURS_EXTENSION})')
        MAX_NEIGHBOURS_EXTENSION = EXACT_NEIGHBOURS_EXTENSION
    else:
      # CONDICION DE SALIDA PRINCIPAL ***
      if paths.empty or abs((total_paths -len(columns))/len(columns))<=0.0002:
        lg.info(f'problem solution')
        break  
      total_paths = len(columns)

      d_v, UB2 = main_rmp(general_params, general_info,columns,LABELING_ITERATION_COUNTER)
      lg.info(f'[{LABELING_ITERATION_COUNTER}]UB2:{UB2}, paths found:{len(paths)} total unique paths: {len(columns)}') 
   
  computation_done.set()
  dump_results()

def dump_results(signum=None, frame=None):
  total_process_time = round(time.perf_counter() - MAIN_START_TIME,2)
  summary_file_name=''
  details_filepath=''
  if len(columns) > 0: 
    columns.sort_values(by=['start','end','type'], inplace=True, ignore_index=True)
    if PROBLEM_TYPE == 'CABDATA':
      summary_file_name = 'CABDATA_CG_path_generation_summary.csv'
      details_filepath =  f'{BASE_PATH}/paths/cg/cabdata'
    else:
      summary_file_name = 'MTL_CG_path_generation_summary.csv'
      details_filepath =  f'{BASE_PATH}/paths/cg/mtl'
    
    os.makedirs(details_filepath, exist_ok=True)
    write_process_summary(general_params, total_process_time, len(columns) - TOTAL_HEURISTIC_PATHS , len(columns), UB1, UB2, summary_file_name )
    write_results_from_paths(general_params, columns.to_numpy(), total_process_time, None,  '_setting', details_filepath )
  sys.exit(0)
  

# Event to signal the end of computation
computation_done = threading.Event()

def timer():
    if not computation_done.wait(MAX_EXECUTION_TIME):
        lg.info("Maximum execution time reached. Interrupting the process.")
        signal.raise_signal(signal.SIGINT)


if __name__ == '__main__':
  general_params = Parameters(
    n=int(ifnull(1, DEFAULT_PARAMS.get('N'))),         # number of nodes, 
    p=int(ifnull(2,DEFAULT_PARAMS.get('P'))),           # target number of hubs
    r=float(ifnull(3,DEFAULT_PARAMS.get('R'))),       # r function param
    v=float(ifnull(4,DEFAULT_PARAMS.get('V'))),       # Access time factor
    alpha=float(ifnull(5, DEFAULT_PARAMS.get('ALPHA'))),  # Discount factor 
    perc=float(ifnull(6,1)),      # percentage of edges used
    epsilon=float(ifnull(7,0.00005)) # minimum reduced profit  
  )
  
  MAX_NEIGHBOURS_EXTENSION = int(ifnull(8, DEFAULT_PARAMS.get('MAX_NEIGHBOURS_EXTENSION')))
  MAX_EXECUTION_TIME = int(ifnull(9, DEFAULT_PARAMS.get('MAX_EXECUTION_TIME')))
  MAX_LABELS_PER_COMODITY = int(ifnull(10, DEFAULT_PARAMS.get('MAX_LABELS_PER_COMODITY')))
  IS_HYBRID = eval(ifnull(11, IS_HYBRID))
  PROBLEM_TYPE = ifnull(12, PROBLEM_TYPE)
  
  # Start the timer thread
  timer_thread = threading.Thread(target=timer)
  timer_thread.start()
  # if the SIGINT signal is thrown, the process writes the results so far and finishes
  signal.signal(signal.SIGINT, dump_results)  
  
  BASE_PATH=get_base_path()
  
  os.makedirs(f"{BASE_PATH}/paths", exist_ok=True)
  os.makedirs(f"{BASE_PATH}/logs", exist_ok=True)
  os.makedirs(f"{BASE_PATH}/results", exist_ok=True)
  
  os.environ.get('LABELING_LOGGING_LEVEL', 'INFO')
  
  # setting the logging config  
  lg.basicConfig(
    filename= f'{BASE_PATH}/logs/labelling_setting-{general_params.n}-{general_params.alpha}-{general_params.v}-{general_params.p}-{general_params.perc}-{general_params.r}-{"{:.5f}".format(general_params.epsilon)}.{PROBLEM_TYPE}.{"hybrid" if IS_HYBRID else "heuristic"}.{datetime.now().strftime("%Y%m%d%H%M%S")}.log',
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=lg.getLevelName(os.environ.get('LABELING_LOGGING_LEVEL', 'INFO')))
  lg.getLogger().addHandler(lg.StreamHandler(sys.stdout))
  
  
  lg.info(f'starting processing paths using n:{general_params.n} alpha:{general_params.alpha} v:{general_params.v} p:{general_params.p} perc:{general_params.perc} r:{general_params.r} epsilon:{general_params.epsilon}')
  general_info = GeneralInfo(general_params, BASE_PATH,suffix=("_mtl" if PROBLEM_TYPE=='MTL' else None))
  #general_info = GeneralInfo(general_params, BASE_PATH,"_mtl")
  MAIN_START_TIME = MAIN_START_TIME = time.perf_counter()
  main_labeling(general_params, general_info, BASE_PATH)
  lg.info(f'finishing processing paths using n:{general_params.n} alpha:{general_params.alpha} v:{general_params.v} p:{general_params.p} perc:{general_params.perc} r:{general_params.r} epsilon:{general_params.epsilon}')
  
