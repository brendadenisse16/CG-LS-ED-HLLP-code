from common.structures.general_info import GeneralInfo
from common.structures.label import Label
from common.structures.parameters import Parameters
import logging as lg
import numpy as np

from common.utils.data_utils import get_hubs, get_path_cost_2


class LabelManager:
  ''' 
    label_catalog is a 4-level dictionary (fixed)
    1. type
    2. source
    3. dest
    4. resource
    The order is not random, as the main search process will query for the first three levels, 
    when evaluating domination
    #
    '''
  def __init__(self, parameters:Parameters, info: GeneralInfo = None):
    self.next_id = 0 # este asignara el ID a cada etiqueta que se cree
    self.parameters = parameters
    # self.label_list = []
    self.label_idx = 0
    self.info = info
    
  def get_next_id(self):
    self.next_id += 1
    return self.next_id
  
  def createLabel(self, node, time, cost, path=None, resource=None, label_type=1, visited:np.array=None):
    path = path = path or [node]
    
    if visited is None:
      visited = np.repeat(0, self.parameters.n)
      visited[path] = 1
    else:
      visited[node] = 1
    
    l = Label(self.label_idx, node, time, cost, path, resource, label_type, visited)
    # self.label_list.append(l)
    self.label_idx += 1
    return l
  
  
  def extend(self, label, node, accum_time, accum_cost, resource, label_type=1) -> Label:
    """returns a new label, based on the extension between the provided label and node

    Args:
        label (Label): label to be extended
        node (int): node to be extended to
        edge_time (int): travel time correspoding to the extended edge
        edge_cost (int): cost correspoding to the extended edge
        resource (int): the consumed resource, i.e., a hub edge
        label_type (int, optional): label type. Defaults to 1.
        visited (list, optional): binary array, 1 if a node was visited in the path, 0 if not. Defaults to None.

    Returns:
        Label: the new label resulting from the extension
    """
    
    if label.visited[node] == 1:
      lg.warning(f'invalid extension, node already visited, path:{label.path}; node:{node}')
      return None
        
    path_new = label.path.copy()
    path_new.append(node)
    return self.createLabel(node, accum_time, accum_cost, path_new, resource, label_type)
  
  
  def get_label(self, id):
    return self.label_list[id]
    