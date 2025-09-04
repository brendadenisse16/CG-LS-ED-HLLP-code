import time
from common.structures.general_info import GeneralInfo
from common.structures.parameters import Parameters
from common.utils.data_utils import get_path_cost_2, get_reverse_path, clean_concat_paths_df
from common.utils.csv_data_utils import write_results_from_paths
import pandas as pd 
import os
import itertools
from collections import defaultdict, deque

from common.utils.data_utils import ifnull

DEFAULT_PARAMS = {
  'N':10,
  'P':5,
  'ALPHA':0.2,
  'V': 0.1,
  'R': 1.7,
  'PROBLEM_TYPE':'CABDATA',
}

PROBLEM_TYPE=DEFAULT_PARAMS.get('PROBLEM_TYPE')

def get_details_df(info:GeneralInfo, problem_type:str):
  header_names = [
    'fecha',
    'formula',
    'n',
    'p',
    'alpha',
    'perc',
    'seed',
    'v',
    'r',
    'total_time',
    'rel_mip_gap',
    'abs_mip_gap',
    'hub_nodes',
    'hub_edges',
    'access_edges',
    'exit_edges',
    'paths_sel'
  ] 
  read_file = f'{info.base_path}/results/cg/{problem_type.lower()}/details.csv'
 
  details_df = pd.read_csv(read_file, sep=";",header=0,names=header_names)  
  details_df.drop_duplicates(subset=['n','p','alpha'], keep='last',ignore_index=True, inplace=True)
  return details_df

def get_result_detail(info:GeneralInfo, n:int, p:int, alpha:float, problem_type:str):
  details_df = get_details_df(info, problem_type)
  filtered_df =  details_df.query(
    f'n == {n} and p == {p} and alpha == {alpha}'
  )
  if not filtered_df.empty:
      # Ensure the DataFrame is sorted (if necessary)
      filtered_df = filtered_df.sort_index()  # Replace with `sort_values('id')` if you have an 'id' column
      # Return the last record as a Series
      return filtered_df.iloc[-1]
  else:
      return None

def find_ordered_line(hub_edges) -> list:
  """
  Reconstructs the ordered line from a list of undirected hub edges.

  Parameters:
  hub_edges (list of tuple): List of undirected edges (a, b).

  Returns:
  list: Ordered line of nodes.
  """

  # Step 1: Create adjacency list
  adjacency = defaultdict(list)
  for a, b in hub_edges:
    adjacency[a].append(b)
    adjacency[b].append(a)

  # Step 2: Find the start node
  # A start node will have exactly one neighbor
  start_node = next(node for node in adjacency if len(adjacency[node]) == 1)

  # Step 3: Reconstruct the ordered line
  ordered_line = []
  visited = set()
  queue = deque([start_node])

  while queue:
    current_node = queue.popleft()
    if current_node not in visited:
      visited.add(current_node)
      ordered_line.append(current_node)
      # Add unvisited neighbors to the queue
      for neighbor in adjacency[current_node]:
        if neighbor not in visited:
          queue.append(neighbor)

  return ordered_line

def get_hub_candidates_by_swap(info:GeneralInfo, initial_hub_line:list):
  new_hub_lines = list()
  combinations = list(itertools.combinations(range(len(initial_hub_line)),2))

  for x, y in combinations:
    new_hub_line = initial_hub_line.copy()
    tmp_val_x = new_hub_line[x]
    new_hub_line[x] = new_hub_line[y]
    new_hub_line[y] = tmp_val_x
    

    new_hub_lines += [new_hub_line]
  
  return new_hub_lines

def generate_new_paths_per_hub_line_1(info:GeneralInfo, hub_nodes:list, visited:set):
  non_hub_nodes = [node for node in range(info.parameters.n) if node not in hub_nodes]
  no_hub_comb = itertools.combinations(non_hub_nodes, 2)

  results = []
  total_profit = 0.0
  
  def is_not_visited(hub_candidate:list, path_type:int)->bool:
    nonlocal visited
    if (frozenset(hub_candidate),path_type) in visited:
      return False
    visited.add((frozenset(hub_candidate),path_type))
    return True

  for l in range(1, len(hub_nodes)):
    for sub_hub_line in itertools.combinations(hub_nodes,l+1):
      sub_hub_line = list(sub_hub_line)
      hub_time = sum([info.get_edge_time(u,v,1) for u, v in zip(sub_hub_line[0:-1], sub_hub_line[1:])])
      
      for non_hub_node in non_hub_nodes:
        # tipo 1 (dos escenarios)
        candidate_line = [non_hub_node] + sub_hub_line
        # if is_not_visited(candidate_line, 1):
        source = candidate_line[0]
        dest = candidate_line[-1]
        hub_access_time = info.get_edge_time(non_hub_node, sub_hub_line[0], 1)
        path_time = hub_access_time + hub_time + info.access_time + info.exit_time
        path_dirct_time = info.get_direct_time(source,dest)
        if path_time < path_dirct_time: 
          simple_cost = get_path_cost_2(info, source, dest, path_time)
          path_cost = path_dirct_time * simple_cost
          total_profit += path_cost
          results += [[candidate_line, sub_hub_line, 1, source, dest, path_time, simple_cost, path_cost]]
          # insertamos el path reverso tambien
          rev_candidate_line = get_reverse_path(candidate_line)
          rev_sub_hub_line = get_reverse_path(sub_hub_line)
          results += [[rev_candidate_line, rev_sub_hub_line, 1, dest, source, path_time, simple_cost, path_cost ]]

        # if is_not_visited(candidate_line, 1):
        candidate_line = sub_hub_line + [non_hub_node]
        source = candidate_line[0]
        dest = candidate_line[-1]
        hub_exit_time = info.get_edge_time(sub_hub_line[-1], non_hub_node, 1)
        path_time = hub_time + hub_exit_time + info.access_time + info.exit_time
        path_dirct_time = info.get_direct_time(source,dest)
        if path_time < path_dirct_time: 
          simple_cost = get_path_cost_2(info, source, dest, path_time)
          path_cost = path_dirct_time * simple_cost
          total_profit += path_cost
          results += [[candidate_line, sub_hub_line, 1, source, dest, path_time, simple_cost, path_cost]]
          rev_candidate_line = get_reverse_path(candidate_line)
          rev_sub_hub_line = get_reverse_path(sub_hub_line)
          results += [[rev_candidate_line, rev_sub_hub_line, 1, dest, source, path_time, simple_cost, path_cost ]]

        # tipo 2
        candidate_line = [non_hub_node] + sub_hub_line
        # if is_not_visited(candidate_line, 2):
        source = candidate_line[0]
        dest = candidate_line[-1]
        access_time = info.get_edge_time(source, sub_hub_line[0], 0)
        path_time = access_time + hub_time + info.access_time + info.exit_time
        path_dirct_time = info.get_direct_time(source,dest)
        if path_time < path_dirct_time: 
          simple_cost = get_path_cost_2(info, source, dest, path_time)
          path_cost = path_dirct_time * simple_cost
          total_profit += path_cost
          results += [[candidate_line, sub_hub_line, 2, source, dest, path_time, simple_cost, path_cost]]
          rev_candidate_line = get_reverse_path(candidate_line)
          rev_sub_hub_line = get_reverse_path(sub_hub_line)
          results += [[rev_candidate_line, rev_sub_hub_line, 3, dest, source, path_time, simple_cost, path_cost ]]
        
        # tipo 3
        candidate_line = sub_hub_line + [non_hub_node]
        # if is_not_visited(candidate_line, 3):
        source = candidate_line[0]
        dest = candidate_line[-1]
        exit_time = info.get_edge_time(sub_hub_line[-1], dest, 0)
        path_time = exit_time + hub_time + info.access_time + info.exit_time
        path_dirct_time = info.get_direct_time(source,dest)
        if path_time < path_dirct_time: 
          simple_cost = get_path_cost_2(info, source, dest, path_time)
          path_cost = path_dirct_time * simple_cost
          total_profit += path_cost
          results += [[candidate_line, sub_hub_line, 3, source, dest, path_time, simple_cost, path_cost]]
          rev_candidate_line = get_reverse_path(candidate_line)
          rev_sub_hub_line = get_reverse_path(sub_hub_line)
          results += [[rev_candidate_line, rev_sub_hub_line, 2, dest, source, path_time, simple_cost, path_cost ]]
      
      # tipo 4
      for ac_node, ex_node in itertools.combinations(non_hub_nodes, 2):
        candidate_line = [ac_node] + sub_hub_line + [ex_node]
        # if is_not_visited(candidate_line, 4):
        source = candidate_line[0]
        dest = candidate_line[-1]
        access_time = info.get_edge_time(source, sub_hub_line[0], 0)
        exit_time = info.get_edge_time(sub_hub_line[-1], dest, 0)
        path_time = access_time + hub_time + exit_time + info.access_time + info.exit_time
        path_dirct_time = info.get_direct_time(source,dest)
        if path_time < path_dirct_time: 
          simple_cost = get_path_cost_2(info, source, dest, path_time)
          path_cost = path_dirct_time * simple_cost
          total_profit += path_cost
          results += [[candidate_line, sub_hub_line, 4, source, dest, path_time, simple_cost, path_cost]]
          rev_candidate_line = get_reverse_path(candidate_line)
          rev_sub_hub_line = get_reverse_path(sub_hub_line)
          results += [[rev_candidate_line, rev_sub_hub_line, 4, dest, source, path_time, simple_cost, path_cost ]]
          
        candidate_line = [ex_node] + sub_hub_line + [ac_node]
        # if is_not_visited(candidate_line, 4):
        source = candidate_line[0]
        dest = candidate_line[-1]
        access_time = info.get_edge_time(source, sub_hub_line[0], 0)
        exit_time = info.get_edge_time(sub_hub_line[-1], dest, 0)
        path_time = access_time + hub_time + exit_time + info.access_time + info.exit_time
        path_dirct_time = info.get_direct_time(source,dest)
        if path_time < path_dirct_time: 
          simple_cost = get_path_cost_2(info, source, dest, path_time)
          path_cost = path_dirct_time * simple_cost
          total_profit += path_cost
          results += [[candidate_line, sub_hub_line, 4, source, dest, path_time, simple_cost, path_cost]]
          rev_candidate_line = get_reverse_path(candidate_line)
          rev_sub_hub_line = get_reverse_path(sub_hub_line)
          results += [[rev_candidate_line, rev_sub_hub_line, 4, dest, source, path_time, simple_cost, path_cost ]]
          
        sub_hub_line.reverse()
        
        candidate_line = [ac_node] + sub_hub_line + [ex_node]
        # if is_not_visited(candidate_line, 4):
        source = candidate_line[0]
        dest = candidate_line[-1]
        access_time = info.get_edge_time(source, sub_hub_line[0], 0)
        exit_time = info.get_edge_time(sub_hub_line[-1], dest, 0)
        path_time = access_time + hub_time + exit_time + info.access_time + info.exit_time
        path_dirct_time = info.get_direct_time(source,dest)
        if path_time < path_dirct_time: 
          simple_cost = get_path_cost_2(info, source, dest, path_time)
          path_cost = path_dirct_time * simple_cost
          total_profit += path_cost
          results += [[candidate_line, sub_hub_line, 4, source, dest, path_time, simple_cost, path_cost]]
          rev_candidate_line = get_reverse_path(candidate_line)
          rev_sub_hub_line = get_reverse_path(sub_hub_line)
          results += [[rev_candidate_line, rev_sub_hub_line, 4, dest, source, path_time, simple_cost, path_cost ]]
          
        candidate_line = [ex_node] + sub_hub_line + [ac_node]
        # if is_not_visited(candidate_line, 4):
        source = candidate_line[0]
        dest = candidate_line[-1]
        access_time = info.get_edge_time(source, sub_hub_line[0], 0)
        exit_time = info.get_edge_time(sub_hub_line[-1], dest, 0)
        path_time = access_time + hub_time + exit_time + info.access_time + info.exit_time
        path_dirct_time = info.get_direct_time(source,dest)
        if path_time < path_dirct_time: 
          simple_cost = get_path_cost_2(info, source, dest, path_time)
          path_cost = path_dirct_time * simple_cost
          total_profit += path_cost
          results += [[candidate_line, sub_hub_line, 4, source, dest, path_time, simple_cost, path_cost]]
          rev_candidate_line = get_reverse_path(candidate_line)
          rev_sub_hub_line = get_reverse_path(sub_hub_line)
          results += [[rev_candidate_line, rev_sub_hub_line, 4, dest, source, path_time, simple_cost, path_cost ]]
        
  return total_profit, results

def open_close(info:GeneralInfo, initial_hub_line:list, initial_profit:float, visited:set):
  
  non_hub_nodes = [node for node in range(info.parameters.n) if node not in initial_hub_line]
  
  n2_profit = initial_profit
  n2_paths = list()
  root_hub_line = initial_hub_line
  
  for idx in range(info.parameters.p):
    for non_hub_node in non_hub_nodes:
      candidate_hub_line = root_hub_line.copy()
      candidate_hub_line[idx] = non_hub_node
      
      n2_candidate_profit, n2_candidate_paths = generate_new_paths_per_hub_line_1(info, candidate_hub_line, visited)
      
      n2_paths += n2_candidate_paths
      if n2_candidate_profit > n2_profit:
        n2_profit = n2_candidate_profit

  
  return (n2_profit, n2_paths)

def get_cg_columns(info:GeneralInfo):
  filepath=''
  if( PROBLEM_TYPE == 'MTL' ):
    filepath = f'{info.base_path}/paths/cg/mtl'
  else:
    filepath = f'{info.base_path}/paths/cg/cabdata'
  read_file=f'{filepath}/data_{info.parameters.n}_{info.parameters.p}_{info.parameters.alpha}_{info.parameters.perc}_{info.parameters.v}_{info.parameters.r}_4_setting.csv'

  current_columns = pd.read_csv(
    read_file,
    sep='\t', 
    header=None, 
    skip_blank_lines=True,
    names=['path','start','end','type','distance','cost'],
    skiprows=lambda x: x < 2
  )

  current_columns['path'] = current_columns['path'].apply(eval)
  return current_columns

def dump_results(info:GeneralInfo, columns:pd.DataFrame, total_process_time):
  if len(columns) <= 0: return 
  
  columns.sort_values(by=['start','end','type'], inplace=True, ignore_index=True)
  data_dir = 'cabdata' if PROBLEM_TYPE == 'CABDATA' else 'mtl'
  
  write_results_from_paths(info.parameters, columns.to_numpy(), total_process_time, None,  '_localsearch', f'{info.base_path}/paths/localsearch/{data_dir}' )

def get_base_path():
  return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if __name__ == '__main__':
  MAIN_START_TIME = time.perf_counter()
  general_params = Parameters(
    n=int(ifnull(1, DEFAULT_PARAMS.get('N'))),         # number of nodes, 
    p=int(ifnull(2,DEFAULT_PARAMS.get('P'))),           # target number of hubs
    r=float(ifnull(3,DEFAULT_PARAMS.get('R'))),       # r function param
    v=float(ifnull(4,DEFAULT_PARAMS.get('V'))),       # Access time factor
    alpha=float(ifnull(5, DEFAULT_PARAMS.get('ALPHA'))),  # Discount factor 
    perc=float(ifnull(6,1)),      # percentage of edges used
    epsilon=float(ifnull(7,0.00005)) # minimum reduced profit  
  )
  PROBLEM_TYPE = ifnull(8, PROBLEM_TYPE)
  
  info = GeneralInfo(general_params, get_base_path(), suffix=("_mtl" if PROBLEM_TYPE=='MTL' else None))

  detail = get_result_detail(info,info.parameters.n,info.parameters.p,info.parameters.alpha, PROBLEM_TYPE)[["hub_nodes", "hub_edges", "paths_sel","abs_mip_gap"]]
  #applying eval to cast the strings as lists
  hub_nodes = eval(detail["hub_nodes"])
  hub_edges = eval(detail["hub_edges"])
  paths_sel_idx = eval(detail["paths_sel"])
  paths_sel_idx = [ x+2 for x in paths_sel_idx]
  solution_profit = detail["abs_mip_gap"]
  visited = set()

  initial_hub_line = find_ordered_line(hub_edges) # <--- PATH RESULTADO DE CG !!!

  new_hub_lines = get_hub_candidates_by_swap(info, initial_hub_line)
  best_profit = solution_profit
  omega = list()
  hubs_to_analize = [initial_hub_line] + new_hub_lines
  for hub_line in hubs_to_analize:
    if len(hub_line) < 2: continue
    n1_profit, n1_paths = generate_new_paths_per_hub_line_1(info, hub_line, visited)
    
    omega += n1_paths
    if n1_profit > best_profit:
      best_profit = n1_profit

  n2_profit, n2_paths = open_close(info, initial_hub_line, best_profit, visited)
  omega += n2_paths

  localsearch_columns = pd.DataFrame(data=omega, columns=['path', 'hubs', 'type', 'start', 'end', 'distance', 'cost', 'red_profit'])  
  localsearch_columns = localsearch_columns[["path", "start", "end", "type", "distance", "cost"]]
  cg_columns = get_cg_columns(info)
  total_columns = clean_concat_paths_df(localsearch_columns, cg_columns)
  
  LOCALSEARCH_PATHS_GENERATION_TIME = time.perf_counter() - MAIN_START_TIME
  
  dump_results(info, total_columns, LOCALSEARCH_PATHS_GENERATION_TIME)
  