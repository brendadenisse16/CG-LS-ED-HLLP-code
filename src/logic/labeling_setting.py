from common.structures.general_info import GeneralInfo
from common.structures.label_manager import LabelManager
from common.structures.results_manager import ResultsManager
from common.structures.label import Label
from logic.rmp import get_dv_value
from collections import deque
import numpy as np
import logging as lg

def evaluate_domination(results_manager: ResultsManager, label: Label, resource:int, path_type:int) -> bool:
  resource = resource or label.resource
  for r in range(1, resource + 1):
    llist = results_manager.getLabelList(label.path[0], label.node, r, path_type)
    if llist.evaluate_domination(label):
      return True
  return False

def is_hub_edge_by_position(
  edge_position:int, # 1 for the first edge
  target_resource:int,
  path_type:int
  ) -> int:
  #
  # type 1: all hub edges
  #
  res = 0
  if path_type == 1:
    res = 1
  #
  # type 2: ORIGIN NO HUB, DESTINATION HUB
  elif path_type == 2:
    if edge_position > 1:
      res = 1
  #
  # type 3: ORIGIN HUB, DESTINATION NO HUB
  elif path_type == 3:
    if edge_position <= target_resource:
      res = 1
    
  # type 4: ORIGIN NO HUB,  DESTINATION NO HUB
  elif path_type == 4:
    if edge_position > 1 and edge_position < (target_resource + 2):
      res = 1
  
  return res

def is_final_edge(edge_position, path_type, path_resource, target_resource):
    is_resource_match = path_resource == target_resource
    #is_destination_match = path_last_node == destination
    is_destination_match = True

    if path_type in (1, 2) and is_resource_match and is_destination_match:
        return True
    elif path_type == 3 and is_resource_match and is_destination_match and edge_position > target_resource:
        return True
    elif path_type == 4 and is_resource_match and is_destination_match and edge_position > target_resource + 1:
        return True
    else:
        return False


def get_valid_extension(info: GeneralInfo, 
                        label: Label,
                        destination: int, 
                        is_hub: bool, 
                        is_final_edge: bool,
                        max_neighbours_extension:int):
    """
    Calculate valid extensions from a given node considering various constraints.

    Args:
        info (GeneralInfo): Information about the graph.
        label_time (int): Current time of the label.
        label_node (int): The current node of the label.
        label_visited (np.array): Array indicating visited nodes.
        destination (int): Destination node.
        is_hub (bool): Indicates if the extension is a hub.
        is_final_edge (bool): Indicates if this is the final edge.
        max_neighbours_extension (int): heuristic parameter.  Oriented to select the N best extensions

    Returns:
        Iterable: Pairs of (node, extended_time) for valid extensions.
    """
    direct_time = np.round(info.time_matrix[label.path[0],destination],3)
    if is_final_edge:
        extended_final_time = round(
            label.time + round(info.time_matrix[label.node, destination],3) - 
            round(is_hub * info.time_matrix[label.node, destination] * (1 - info.parameters.alpha), 3)
            ,3)
        if extended_final_time < direct_time:
            return([(destination, extended_final_time)])
        else:
            return ([])
    
    neighbor_times = np.round(info.get_edge_neigbours_times(label.node),3)
    extended_times = np.round(label.time + neighbor_times - np.round(is_hub * neighbor_times * (1 - info.parameters.alpha), 3),3)
    valid_distance_extension = extended_times < direct_time
    
    unvisited = [not(bool(i)) for i in label.visited]  # Directly using NumPy array for boolean negation
    valid_extension = valid_distance_extension & unvisited

    last_edge_filter = ~np.full(len(neighbor_times), is_final_edge)
    last_edge_filter[destination] = not last_edge_filter[destination]
    valid_extension &= last_edge_filter  # In-place bitwise AND operation

    result = list(zip(np.where(valid_extension)[0], np.round(extended_times[valid_extension],3)))

    return result[:max_neighbours_extension]
  

def get_labels_by_commodity(
  info:GeneralInfo,
  source:int,
  destination:int,
  dual_values:dict,
  lm: LabelManager,
  rm: ResultsManager,
  path_type:int=1,
  resource: int=1,
  max_neighbours_extension:int=200
):
  Q = deque()
  in_out_time = round(info.access_time + info.exit_time, 3)
  l_0 = lm.createLabel(source, in_out_time, 0, [source], 0, path_type)
  Q.append(l_0)

  while Q:
    l = Q.pop()
    #neighbors = info.get_neighbours(l.node)
    is_hub_edge = is_hub_edge_by_position(len(l.path), resource, path_type) # 0 o 1
    extended_resource = l.resource + is_hub_edge
    is_f_edge = is_final_edge(len(l.path), path_type, extended_resource, resource)

    for neighbor, extended_time in get_valid_extension(info, l, destination, is_hub_edge, is_f_edge,max_neighbours_extension):
      # Compute the extended cost
      extended_cost = l.cost
      if is_hub_edge > 0: #solo si es hub edge
        f_dual_value = get_dv_value(dual_values, 'F', source, destination, l.node, neighbor)
        if f_dual_value is None:
            f_dual_value = 0
        
        extended_cost = extended_cost + f_dual_value
        
      # Extend the label and process domination
      l_l = lm.extend(
        l, 
        neighbor,
        extended_time,
        extended_cost,
        extended_resource,
        path_type)
      if not is_f_edge and (l_l is None or evaluate_domination(rm, l_l, l_l.resource, l_l.type)):
        l_l = None
        continue
      
      # Add to queue or result manager
      if is_f_edge and l_l.node == destination:
        yield l_l
        # llist = rm.getLabelList(source, neighbor, l_l.resource, l_l.type)
        # llist.add_by_cost_order_desc(l_l)  #### S E T T I N G
      else:
        Q.append(l_l)
 

def generate_paths_for_destination(info:GeneralInfo, source:int, dual_values:dict, lm, rm, ttype, max_neighbours_extension:int, max_labels_per_comodity:int):
  parameters = info.parameters
  for v in range(parameters.n): # destination
    labels_found = 0
    if source >= v: continue
    for r in range(1, (parameters.p)): # resource  
      lg.debug(f'processing commodity: ({source},{v}), path type: {ttype}, resource: {r}')
      for label in get_labels_by_commodity(
        info=info, 
        source=source, 
        destination=v, 
        dual_values=dual_values, 
        lm=lm, 
        rm=rm, 
        path_type=ttype, 
        resource=r,
        max_neighbours_extension=max_neighbours_extension
        ):
        # validate if the path has a new node
        labels_found += rm.add_setting_label(label)
        if labels_found >= max_labels_per_comodity: return
      

         
def generate_paths(dual_values:dict, info: GeneralInfo, max_neighbours_extension:int, max_labels_per_comodity:int):
  lg.debug('Starting Labeling Path Generation')
  
  parameters = info.parameters
  lm = LabelManager(parameters, info)
  rm = ResultsManager(parameters)

  for ttype in range(1, 5): # path type
    for u in range(parameters.n): # sourceq
      generate_paths_for_destination(info, u, dual_values,lm,rm,ttype,max_neighbours_extension, max_labels_per_comodity)
      # for v in range(parameters.n): # destination
      #   if u >= v: continue
      #   for r in range(1, (parameters.p)): # resource  
      #     lg.debug(f'processing commodity: ({u},{v}), path type: {ttype}, resource: {r}')
      #     get_labels_by_commodity(
      #       info=info, 
      #       source=u, 
      #       destination=v, 
      #       dual_values=dual_values, 
      #       lm=lm, 
      #       rm=rm, 
      #       path_type=ttype, 
      #       resource=r,
      #       max_neighbours_extension=max_neighbours_extension
      #       )
          
  
  return rm.get_result_df(info, dual_values)
  