import numpy as np
import pandas as pd
import networkx as nx
from common.structures.parameters import Parameters
from common.structures.general_info import GeneralInfo

from common.utils.utils import get_hubs,get_path_type,random_choice_noreplace_vector,get_path_bin_sum_from_bin,get_direct_time_dict, get_path_bin, get_path_bin_sum, get_selected_hub_nodes_binary_vector, get_selected_hub_edges_binary_matrix, get_hub_node_time, get_total_profit, get_path_cost
from logic.shortest_paths import shortest_simple_paths_mod
from common.utils.data_utils import get_path_cost_2



def get_path_type(path):
    has_ac_node, has_ex_node = path[0]!=path[1], path[-1]!=path[-2]
    type=1 
    if has_ac_node and not has_ex_node:
        type=2
    elif has_ex_node and not has_ac_node:
        type=3
    elif has_ac_node and has_ex_node:
        type=4

    return type

def clean_path(path):
    clean_access, clean_exit = path[0]==path[1], path[-1]==path[-2]
    res_path=path
    if clean_access:
        res_path=path[1:]
    if clean_exit:
        res_path=res_path[:-1]
    return res_path

def reverse_type(type):
  cases = {
    1:1,
    2:3,
    3:2,
    4:4
  }
  return cases.get(type,type)

def generate_heuristic_diGraph(parameters:Parameters, info:GeneralInfo) -> nx.DiGraph:
  G = nx.DiGraph()  

  all_nodes = [ str(node) for node in range(parameters.n)]
  all_nodes = all_nodes + ['h'+str(edge) for edge in range(parameters.n)]
  G.add_nodes_from(all_nodes)

  for i in range(parameters.n):
    for j in range(parameters.n):
      if i > j:
        # adding the hub edges
        source = 'h'+str(i)
        dest = 'h'+str(j)
        w = parameters.alpha * info.time_matrix[i,j]
        G.add_edge(source, dest, weight=w)
        G.add_edge(dest, source, weight=w)
        
        # adding the direct edges
        source = str(i)
        dest = str(j)
        w = info.time_matrix[i,j]
        G.add_edge(source, dest, weight=w)
        G.add_edge(dest, source, weight=w)
        
        # adding access/exit edges
        G.add_edge('h' + source, dest, weight=w)
        G.add_edge(source, 'h' +dest, weight=w)
        G.add_edge('h' + dest, source, weight=w)
        G.add_edge(dest, 'h' + source, weight=w)
        
    # artificial edges
    source = str(i)
    dest = 'h' + source
    w = 0.0
    G.add_edge(source, dest,weight=w)
    G.add_edge(dest, source,weight=w)    
  return G

def _numeric_node(str_node):
  return int(str_node.replace('h', ''))

def _numeric_path(str_path_list):
  return list(map(_numeric_node, str_path_list))


def get_heuristic_paths(parameters:Parameters, info:GeneralInfo, max_heuristic_iter, max_paths_per_comodity) -> pd.DataFrame:
  G = generate_heuristic_diGraph(parameters, info)
  res_paths = list()
  access_exit_time = info.access_time + info.exit_time

  for u in range(parameters.n):
    for v in range(parameters.n):
      if u >= v: continue
      dt = info.time_matrix[u,v]
      counter = 0
      added_paths_counter = [0,0,0,0] 
      for path in nx.shortest_simple_paths(G, str(u), str(v), weight='weight'):
        if counter >= max_heuristic_iter:
          break
        counter += 1
        path_time = round(sum([G.adj[i][j]['weight'] for i, j in zip(path[0:-1], path[1:]) ]) + access_exit_time, 3)
        
        if path_time >= dt:  
          break
        
        inner_line = path[1:-1]
        valid_hub = len(inner_line)>1 and np.all([char[0] == 'h' for char in inner_line])
        
        if not valid_hub: # valido la continuidad de la linea hub
          continue
        
        numeric_path = _numeric_path(path)
        hubs = numeric_path[1:-1]
        
        if len(hubs) > parameters.p:
          continue

        path_type = get_path_type(numeric_path)
        
        numeric_path = clean_path(numeric_path) 
        if len(set(numeric_path)) < len(numeric_path):
          continue 

        path_cost = get_path_cost_2(info, numeric_path[0], numeric_path[-1], path_time)
        
        if added_paths_counter[path_type] >= max_paths_per_comodity:
          continue
        
        #res_paths.append([numeric_path, hubs, path_type, numeric_path[0], numeric_path[-1], path_time, path_cost])
        res_paths.append([numeric_path, numeric_path[0], numeric_path[-1], path_type, path_time, path_cost])
        
        rev_path = numeric_path.copy()
        rev_path.reverse()
        
        # rev_hubs = hubs.copy()
        # rev_hubs.reverse()
        
        rev_type = reverse_type(path_type)
        
        #res_paths.append([rev_path, rev_hubs, rev_type, rev_path[0], rev_path[-1], path_time, path_cost])
        res_paths.append([rev_path, rev_path[0], rev_path[-1], rev_type, path_time, path_cost])
        
        added_paths_counter[path_type-1] +=1
                                                                                      
  #res_df = pd.DataFrame(res_paths, columns=['path', 'hubs', 'type', 'start', 'end', 'distance', 'cost'])
  res_df = pd.DataFrame(res_paths, columns=['path', 'start', 'end', 'type', 'distance', 'cost'])
        
  return pd.DataFrame.from_dict(res_df)
