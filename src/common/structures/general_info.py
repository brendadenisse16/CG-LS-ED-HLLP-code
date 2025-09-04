

from common.utils.csv_data_utils import get_time_and_demand_matrices
from common.utils.data_utils import get_node_feature, get_direct_time_dict, get_access_time
import numpy as np
import os

class GeneralInfo:
  def __init__(self, parameters, base_path=os.getcwd(), suffix='') -> None:
    """Initializes general information based on provided parameters.
     :param parameters: The parameters object that includes required constants.
      """
    self.time_matrix, self.demand_matrix = get_time_and_demand_matrices(parameters.n, base_path, suffix=suffix)
    self.node_feature_vector = get_node_feature(parameters.n, self.demand_matrix)
    self.direct_times = get_direct_time_dict(self.time_matrix)
    self.access_time = get_access_time(self.time_matrix, parameters.n, parameters.v)
    self.exit_time = self.access_time
    self.parameters = parameters
    self.base_path = base_path
        
  def get_neighbours(self, node):
    """
    Returns the valid neighbours (direct time != 0)
    """
    return np.where(self.time_matrix[node,] > 0)[0]
    #return self.time_matrix[node][self.time_matrix[node] != 0]

  def get_direct_time(self, u, v):
      return self.time_matrix[u,v]
    
  def get_edge_time(self, u, v, is_hub:int):
    assert u != v, "GeneralInfo.get_edge_time: u and v should be different"
    t = self.get_direct_time(u, v)
    time_saving = round(is_hub * t * (1 - self.parameters.alpha),3)
    return round(t - time_saving, 3)
    
  def get_edge_neigbours_times(self, node):
    return self.time_matrix[node,]
  