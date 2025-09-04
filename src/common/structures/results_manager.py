from common.structures import LinkedLabelList
from common.structures.LinkedLabelList import LinkedList, Node
from common.structures.general_info import GeneralInfo
from common.structures.label_manager import LabelManager
from common.structures.parameters import Parameters
import pandas as pd
import logging as lg
import numpy as np

from logic.rmp import get_total_dual_values

class ResultsManager:
  def __init__(self, params:Parameters):
    self.params = params
    self.results = self.__initialize_results()
    self.visited_idx = self.__initialize_visited_index()
    
  def __initialize_results(self):
    # the comodities will be represented as a 2D matrix
    return(
      [[[[LinkedList() 
        for v in range(self.params.n)]      # destination
        for u in range(self.params.n)]      # origin
        for r in range(self.params.p - 1)]  # resource
        for t in range(4)])                 # type
  
  def __initialize_visited_index(self):
    return(
      #generate a numpy vector of N dimension, all 0s
      {(u, v, t): np.repeat(0,self.params.n)
      for u in range(self.params.n)
      for v in range(self.params.n)
      for t in range(4)}
    )
  
  def __get_visited_vector_by_comodity_by_type(self, source, destination, ltype):
    return self.visited_idx.get((source, destination, ltype-1))
  
  def __set_visited_vector_by_commodity_by_type(self, source, destination, ltype, visited_vector):
    self.visited_idx[(source, destination, ltype-1)] = visited_vector
  
  def addLabelId(self, label, linked_list: LinkedList, insert_before_id:int):
    linked_list.add_before(insert_before_id, Node(label.id, label))
    
  
  def getLabelList(self, u,v, resource, path_type) -> LinkedLabelList:
    #return self.results[u][v][resource-1]
    return self.results[path_type-1][resource-1][u][v]

  def get_result_df(self, info: GeneralInfo, dual_values:dict):
    data = [label_node.label.to_result(info) 
            for typ_l in self.results 
            for res_l in typ_l 
            for or_l in res_l 
            for dest_l in or_l
            for label_node in dest_l
            ]
    res_df = pd.DataFrame(data=data, columns=['path', 'hubs', 'type', 'start', 'end', 'distance', 'cost', 'red_profit'])
    
    if len(res_df) > 0:
      try:
        res_df['dv_cols'] = res_df[['path','hubs','type']].apply(lambda row: get_total_dual_values(row['path'], row['hubs'], row['type'], dual_values), axis=1)
        res_df[['dvA','dvB','dvC','dvD','dvE','dvF','dvG','dvH','dvI']] = res_df.dv_cols.to_list()
        res_df.drop(columns=['dv_cols'],inplace=True)
        res_df['cost2'] = res_df[['start','end','cost']].apply(lambda r: round(info.time_matrix[int(r.start),int(r.end)]*r.cost, 3), axis=1)
        #res_df['red_profit'] = res_df['cost2'] - (res_df['dvA'] + res_df['dvB'] + res_df['dvC'] + res_df['dvF'] + res_df['dvH'] + res_df['dvI']) 
        res_df['red_profit'] = res_df['cost2'] - (res_df['dvC'] + res_df['dvF'] + res_df['dvH'] + res_df['dvI']) 
        res_df['is_valid'] = res_df['red_profit'] > self.params.epsilon

        res_df.drop(columns=['dvA','dvB','dvC','dvD','dvE','dvF','dvG','dvH','dvI','is_valid'],inplace=True)
      except Exception as error:
        lg.info(f'res_df shape: {res_df.shape}')
        lg.error('exception occureed: ', type(error).__name__, '-', error)
      
    ## aqui necesito procesar cada path para aplicarle la formula dependiendo de los valores duales en dual_values
    res_df = res_df[res_df['red_profit'] > self.params.epsilon]
   
    # generamos los caminos reversos de los seleccionados
    rev_df = res_df.apply(lambda row: self.__rev_row(row), axis=1)
    
    # arreglamos formato para coincidir con caminos
    res_df = pd.concat([res_df, rev_df], ignore_index=True)
    res_df.drop(columns=['red_profit','cost2'],inplace=True)
    res_df = res_df[['path','start','end','type','distance','cost']].copy()
    
    return res_df
  
  def add_setting_label(self, label):
    source = label.path[0]
    destination = label.path[-1]
    ltype =  label.type
    
    l_visited_vector = self.__get_visited_vector_by_comodity_by_type(source, destination, ltype)
    tmp_vector_sum = l_visited_vector + label.visited
    if not(np.any(tmp_vector_sum == 1)):
      ## there's a newly considered node in the path
      lg.debug(f'label {label} was filtered by the path diversification control')
      return 0   
    
    self.__set_visited_vector_by_commodity_by_type(source, destination, ltype, tmp_vector_sum)
        
    llist = self.getLabelList(source, destination, label.resource, ltype)
    #llist.add_by_cost_order_desc(label)
    llist.add_by_cost_order_asc(label)
    return 1
    # here we can create a parallel directory by source, destination and type
    # we only need to store a numpy array of the sum of the visited vectors
    
  def __rev_row(self,row):
    rev_path_types = {2:3,3:2}
    rev_path = row.path.copy()
    rev_path.reverse()
    rev_hubs = row.hubs.copy()
    rev_hubs.reverse()
    row.path = rev_path
    row.hubs = rev_hubs
    row.type = rev_path_types.get(row.type, row.type)
    row.start = rev_path[0]
    row.end = rev_path[-1]
    return row
    