import time
import cplex
import pandas as pd
import numpy as np
from common.structures.parameters import Parameters
from common.structures.general_info import GeneralInfo
import logging as lg
import json
from common.utils.data_utils import get_hubs, get_path_cost_2
#from common.utils import get_hubs

BASE_PATH='C:/Users/Brenda/Documents/0.SecondPaper/3.Code/3_label_setting_3'

constr_vars = {
    'A':{'enabled': True },
    'B':{'enabled': True },
    'C':{'enabled': True },
    'D':{'enabled': True },
    'E':{'enabled': True },
    'F':{'enabled': True },
    'G':{'enabled': True },
    'H':{'enabled': True },
    'I':{'enabled': True }
}

def get_comodities_indices(general_params:Parameters):
    n = general_params.n
    index_i = np.arange(n*(n-1)/2);
    index_j=np.arange(n*(n-1)/2);
    index_e=np.arange(n*n).reshape(n,n); 
    e=0
    for i in range(0,n):
        for j in range(0,n):
            if i<j:
                index_i[e]=i
                index_j[e]=j
                index_e[i][j]=e
                index_e[j][i]=e
                e=e+1
    
    return index_i, index_j, index_e

def get_start_and_end_coords(init:np.ndarray, end:np.ndarray):
    used_edges = pd.Series(zip(init,end)).drop_duplicates()
    return zip(*used_edges)
    
def get_start_and_end_positions(paths_df:pd.DataFrame):
    df=paths_df.groupby(['start','end']).size().reset_index(name='Sizes')
    position_fin = df['Sizes'].cumsum().to_numpy()
    position_ini = np.append(0, position_fin[:-1])

    return (position_ini, position_fin)

def get_H(aux1:int, nedges:int, paths_df:pd.DataFrame, index_e:np.ndarray):
    H=np.zeros((aux1,nedges))
    hubs = paths_df['hubs'].to_numpy()
    for idx, hub in enumerate(hubs):
        for h_ini, h_fin in zip(hub[:-1],hub[1:]):
            H[idx, index_e[h_ini, h_fin]]=1

    return H


def get_cost_vector(general_info:GeneralInfo, paths_df:pd.DataFrame):
    #np.random.seed(202306)
    #gamma=np.random.rand(len(paths_df))
    direct_times = paths_df['path'].apply(lambda path: general_info.time_matrix[path[0], path[-1]])
    #costs = paths_df['cost']
    costs = paths_df[['path','distance']].apply(lambda row:get_path_cost_2(
        general_info,
        row.path[0], 
        row.path[-1],
        row.distance),axis=1)
    
    #res = (1 + gamma) * direct_times * costs
    res = direct_times * costs

    return res.to_numpy()


def store_dual_value(dv_dict:dict, var_name:str, value: float, c_source=None, c_dest=None, e_source=None, e_dest=None):    
    # variable is enabled
    if constr_vars[var_name] and constr_vars[var_name]['enabled']:
        if var_name not in dv_dict:
            dv_dict[var_name] = dict()
        
        if c_source is None and c_dest is None and e_source is None and e_dest is None:
            dv_dict[var_name] = value
            return dv_dict
        
        elif c_source is not None and c_dest is None and e_source is None and e_dest is None:
            dv_dict[var_name][c_source] = value
            return dv_dict
        
        elif c_source is not None and c_dest is not None and e_source is None and e_dest is None:
            if c_source not in dv_dict[var_name]:
                dv_dict[var_name][c_source] = dict()
            
            dv_dict[var_name][c_source][c_dest] = value
            return dv_dict
        
        elif c_source is not None and c_dest is not None and e_source is not None and e_dest is not None:
            if c_source not in dv_dict[var_name]:
                dv_dict[var_name][c_source] = dict()
            
            if c_dest not in dv_dict[var_name][c_source]:
                dv_dict[var_name][c_source][c_dest] = dict()
                
            if e_source not in dv_dict[var_name][c_source][c_dest]:
                dv_dict[var_name][c_source][c_dest][e_source] = dict()
            
            dv_dict[var_name][c_source][c_dest][e_source][e_dest] = value
            return dv_dict                
            
        else:
            raise Exception("Not supported dual value scenario")


def get_dv_value(dv_dict:dict, var_name:str, c_source=None, c_dest=None, e_source=None, e_dest=None):    
    # variable is enabled
    c_source = str(c_source) if c_source is not None else None
    c_dest =   str(c_dest) if c_dest is not None else None
    e_source =  str(e_source) if e_source is not None else None
    e_dest =  str(e_dest) if e_dest is not None else None
    
    def is_dict(var): isinstance(var,dict)
    
    res = None
    if dv_dict.get(var_name) is not None and constr_vars.get(var_name) is not None and constr_vars.get(var_name).get('enabled'):        
        # simple dual value
        if c_source is None and c_dest is None and e_source is None and e_dest is None:
            res = dv_dict.get(var_name)
        
        # single node dual value
        elif c_source is not None and c_dest is None and e_source is None and e_dest is None:            
            res = dv_dict.get(var_name).get(c_source)
        # comodity node dual value
        elif c_source is not None and c_dest is not None and e_source is None and e_dest is None:
            if dv_dict.get(var_name).get(c_source) is not None:
                res = dv_dict.get(var_name).get(c_source).get(c_dest)
        
        # comodity + edge dual value
        elif c_source is not None and c_dest is not None and e_source is not None and e_dest is not None:
            if dv_dict.get(var_name).get(c_source) is not None:
                if dv_dict.get(var_name).get(c_source).get(c_dest) is not None:
                    if dv_dict.get(var_name).get(c_source).get(c_dest).get(e_source) is not None:
                        res = dv_dict.get(var_name).get(c_source).get(c_dest).get(e_source).get(e_dest)             
            
        else:
            raise Exception("Not supported dual value scenario")
    
    if res is None:
        if var_name =="F":
            res = -99999999999999
        else:
            res = 0
    return round(res,3)


def get_total_dual_values(path:list, hub_nodes:list, type:int, dual_values:dict):
    total_dual_value = 0
    c_source = path[0]
    c_dest = path[-1]

    # static values
    A_val =  get_dv_value(dual_values, "A") or 0
    B_val = get_dv_value(dual_values, "B") or 0
    # single node (on hub nodes)
    C_val = sum(
        [(get_dv_value(dual_values, "C", node) or 0) for node in hub_nodes]
    )

    # comodity
    D_val = get_dv_value(dual_values, "D", c_source, c_dest) or 0
    E_val = get_dv_value(dual_values, "E", c_source, c_dest) or 0
    G_val = get_dv_value(dual_values, "G", c_source, c_dest) or 0
    H_val = get_dv_value(dual_values, "H", c_source, c_dest) or 0
    I_val = get_dv_value(dual_values, "I", c_source, c_dest) or 0

    # comodity + edge
    F_val = sum(
        [
            (get_dv_value(dual_values, "F", c_source, c_dest, e_source, e_dest) or 0)
            for e_source, e_dest in zip(hub_nodes[:-1], hub_nodes[1:])
        ]
    )
    #print(f'F: {F_val}')

    total_dual_value = A_val + B_val + C_val + D_val + E_val + F_val - G_val + H_val + I_val
    #print(f'{A_val}-{B_val}-{C_val}-{D_val}-{E_val}-{F_val}-{G_val}-{H_val}-{I_val}')
    return [A_val, B_val, C_val, D_val, E_val, F_val, G_val, H_val, I_val]
    # return total_dual_value


def main_rmp(general_params:Parameters, general_info:GeneralInfo, paths_info:pd.DataFrame, indice):
    '''
    Will generate the dual values from a group of paths
    Parameters:
    general_params: general problem parameters
    general_info: general information structures (like time and demand matrices)
    path_info: The paths to be consumed by the solver. Dictionary of the following form:
    {
        'path':[[1,2,3], [1,2,3,4,5],[5,3,2]],
        'hubs':[[2,3],[2,3,4,5],[3,2]],
        'type':[3,1,2],
        'start':[1,1,5],
        'end':[3,5,2],
        'distance':[232.23,1234.3234,3245,21],
        'cost':[6535235.424, 5353536.34, 88424234,43]
    }

    '''
    #rsp_start_time = time.perf_counter()
    lg.debug('starting main_rsp...')
    n = general_params.n
    p = general_params.p
    #alpha = general_params.alpha    
    res_dict = dict()
    A = general_info.time_matrix
    B = general_info.demand_matrix
    #node_features_list = general_info.node_feature_vector

    paths_df = paths_info.copy()
    paths_df['hubs'] = paths_df[['path','type']].apply(lambda r: get_hubs(r.path, r.type),axis=1)
    
    index_i = np.arange(n*(n-1)/2);
    index_j=np.arange(n*(n-1)/2);
    index_e=np.arange(n*n).reshape(n,n); 
    e=0
    for i in range(0,n):
        for j in range(0,n):
            if i<j:
                index_i[e]=i;
                index_j[e]=j;
                index_e[i][j]=e;
                index_e[j][i]=e;
                e=e+1;

    ##------------------------------------------------------------------------##
    nedges=int(n*(n-1)/2)
    #number_of_edges=len(edges)
    aux1=paths_df['path'].count()
    #path_time=eval(path_generation_time)
    init = paths_df['start'].to_numpy()
    end = paths_df['end'].to_numpy()
    typ = paths_df['type'].to_numpy()
    # cost = paths_df['cost'].to_numpy()
    # dist = paths_df['distance'].to_numpy()

    ##----------------------------------------------------------------

    # node_features_list=np.zeros(n)
    # for i in range(0,n):
    #     for j in range(0,n):
    #         if i!=j:
    #             node_features_list[i]+=B[i][j]+B[j][i]
    # node_features_list[i]=round(node_features_list[i],3)


    ##------------------------------------------------------------------------##
    tst1 = pd.Series(zip(init,end))
    df=paths_df.groupby(['start','end']).size().reset_index(name='Sizes') #df que contien una start and end nodes y el numero de paths
    # en cada par de o/d nodos
    long=len(df) #len del df con el set de commodities
    auxiliar=df['Sizes'].to_numpy() # numero de paths por cada comoditie
    position_fin=auxiliar.cumsum() # va sumando la cantidad de paths 
    position_ini=np.append(0, position_fin[:-1])


    ##------------------------------------------------------------------------##
    used_edges=tst1.drop_duplicates()
    start_i, start_j = zip(*used_edges)

    H=np.zeros((aux1,nedges))
    hubs = paths_df['hubs'].to_numpy()
    
    for idx, hub in enumerate(hubs):
        for h_ini, h_fin in zip(hub[:-1],hub[1:]):
            H[idx, index_e[h_ini, h_fin]]=1

    # #---------- Index i, index j, ----------------------#
    idx_i = np.arange(n*(n-1))#/2);
    idx_j=np.arange(n*(n-1))#/2);
    e=0
    for i in range(0,n):
        for j in range(0,n):
            if i!=j:
                idx_i[e]=i;
                idx_j[e]=j;
                e=e+1;

    paths_df['cost2'] = paths_df[['path', 'cost']].apply(lambda row: A[row.path[0],row.path[-1]] * row.cost, axis=1)
    cost2 = paths_df['cost2'].to_numpy()

    cpx = cplex.Cplex()
    cpx.set_log_stream(None)
    cpx.set_error_stream(None)
    cpx.set_warning_stream(None)
    cpx.set_results_stream(None)
    cpx.parameters.timelimit.set(900)
    cpx.parameters.threads.set(4)
    cpx.objective.set_sense(cpx.objective.sense.maximize)

    # ADD VARIABLES

    v_var=list(
        cpx.variables.add(
            obj=cost2,
            lb=[0.0]*aux1,
            ub=[cplex.infinity]*aux1,
            names=['v' + str(k) for k in range(aux1)]
        ))

    y_var=[cpx.variables.add(obj=[0 for j in range(0,n)],
                            lb=[0.0]*n,
                            ub=[cplex.infinity]*n,
                            
                            names=['y_(%d)(%d)' % (i, j) for j in range(0,n)]) for i in range(0,n)]


    sume=np.zeros(nedges)


    for e in range(0, nedges):
        sume[e]=np.sum(H[0:aux1,e:e+1])


    for e in range(0,nedges):
        if sume[e]<0.5:
            cpx.variables.set_upper_bounds(y_var[int(index_i[e])][int(index_j[e])],0)
            cpx.variables.set_upper_bounds(y_var[int(index_j[e])][int(index_i[e])],0)
    for i in range(0,n):
        cpx.variables.set_upper_bounds(y_var[i][i],0)
        
    z_var=list(cpx.variables.add(obj=[0]*n,
                                lb=[0.0]*n,
                                ub=[cplex.infinity]*n,
                                #ub=[1]*n,
                                names=['z'+str(i) for i in range(0,n)]))

    l_var=list(
        cpx.variables.add(
            obj=[0]*n,
            lb=[0]*n,
            ub=[cplex.infinity]*n,
            names=['l_'+str(i) for i in range(0,n)]
        ))

    # CONSTRAINTS

    #-----------------------------------------------# 
    # Constraint 8 : ensure that p hubs are opened
    #-------sum_{i=1}^n (z_i)=p---------#
    thevars=[]
    thecoefs=[]
    for i in range(0,n):
        thevars.append(z_var[i])
        thecoefs.append(1)
        
    cpx.linear_constraints.add(
            lin_expr=[cplex.SparsePair(ind = thevars,val = thecoefs)],
            senses=['E'],
            rhs=[p],
            names=['A'])

    # Constrainst 9: establish that p-1 hub edges must be opened 
    # #-------sum_{i!=j} y_{ij}=p-1-----##
    thevars = []
    thecoefs=[]
    for i in range(0,n):
        for j in range(0,n):
            if i!=j:
                thevars.append(y_var[i][j])
                thecoefs.append(1)
                
    cpx.linear_constraints.add(
            lin_expr=[cplex.SparsePair(ind =thevars, val = thecoefs)],
            senses=['E'],
            rhs=[p-1],
            names=['B'])


    # Constraints 27: is to obtain a line oriented to the node
    # with the biggest label number 
    # #---------sum_{i:i!=j} y_{ij}+y_{ji}<=2 z_j-------------#
    for j in range(0,n):
        thevars=[]
        thecoefs = []
        for i in range(0,n):
            if i!=j:
                thevars.append(y_var[j][i])
                thecoefs.append(1.0)
                thevars.append(y_var[i][j])
                thecoefs.append(1.0)
        thevars.append(z_var[j])
        thecoefs.append(-2)        
            
        cpx.linear_constraints.add(
            lin_expr=[cplex.SparsePair(ind=thevars, val=thecoefs)],
            senses=['L'],
            rhs=[0],
            names=['C_'+str(j)])
        


    #---------------------------------
    # Miller-Tucker Zemlin (MTZ) 
    # Constraint 25: is to take advantage of providing
    # an orientation to the hub line structure although
    # the commodities can be supplied in both directions
    #--------- l_i-l_j+n y_{ij}<=n-1 -----------------#   
    for i in range(0,n):
        for j in range(0,n):
            thevars=[]
            thecoefs = []
            if i!=j:
                thevars.append(l_var[i])
                thecoefs.append(1)
                thevars.append(l_var[j])
                thecoefs.append(-1)
                thevars.append(y_var[i][j])
                thecoefs.append(n)
                cpx.linear_constraints.add(
                    lin_expr=[cplex.SparsePair(ind=thevars, val=thecoefs)],
                    senses=['L'],
                    rhs=[n-1],
                    names=['D_'+str(i)+'_'+str(j)])
                

    # Constraint 36: restrict that each commodity is transported either by respectively,
    # using the hub line or taking direct path
    #--------- sum_{k\in r}v_k+e_r=1  -----------------#  
    #sum(v_var[k] for k in range(position_ini[e], position_fin[e])) <= 1
    for e in range(0,long):
        thevars=[]
        thecoefs=[]
        for k in range(int(position_ini[e]),int(position_fin[e])):
            thevars.append(v_var[k])
            thecoefs.append(1)
  
        cpx.linear_constraints.add(
                    lin_expr=[cplex.SparsePair(ind=thevars, val=thecoefs)],
                    senses=['L'],
                    rhs=[1],
                    names=['E_'+str(int(start_i[e]))+'_'+str(int(start_j[e]))])
        

    # Constraint: ensure that commodities can only use paths whose hubs arcs are opened    

    for e in range(0,nedges):
        for ij in range(0,long):
            thevars=[]
            thecoefs=[]
            for k in range(int(position_ini[ij]),int(position_fin[ij])):
                thevars.append(v_var[k])
                thecoefs.append(H[k][e])
            thevars.append(y_var[int(index_i[e])][int(index_j[e])])
            thecoefs.append(-1)
            thevars.append(y_var[int(index_j[e])][int(index_i[e])])
            thecoefs.append(-1)
            cpx.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=thevars, val=thecoefs)],
                        senses=['L'],
                        rhs=[0],
                        names=['F_'+str(int(start_i[ij]))+'_'+str(int(start_j[ij]))+'_'+str(int(index_i[e]))+'_'+str(int(index_j[e]))])


    # Constraint  : to obtain a line oriented to the  
    ##--------sum_{j:i!=j} y_{ij}>=z_i+z_k-1 i=1,...,n-1; k=i+1,...,n
    ##--------RESTRICCION 17-------------------------------------##
    for i in range(0,n-1):
        for k in range(i+1,n):
            thevars=[]
            thecoefs = []
            for j in range(0,n):
                if i!=j:
                    thevars.append(y_var[i][j])
                    thecoefs.append(1)
            thevars.append(z_var[i])
            thecoefs.append(-1)
            thevars.append(z_var[k])
            thecoefs.append(-1)
            
            cpx.linear_constraints.add(
            lin_expr=[cplex.SparsePair(ind=thevars, val=thecoefs)],
            senses=['G'],
            rhs=[-1],
            names=['G_'+str(i)+'_'+str(k)])
            
    #-------------------------------------------------#

    # Valid Inequalities

    for e in range(0,long):
        thevars=[]
        thecoefs=[]
        for k in range(int(position_ini[e]),int(position_fin[e])):
            if typ[k]<3:
                thevars.append(v_var[k])
                thecoefs.append(1)
        thevars.append(z_var[int(start_j[e])])
        thecoefs.append(-1)
        cpx.linear_constraints.add(
                    lin_expr=[cplex.SparsePair(ind=thevars, val=thecoefs)],
                    senses=['L'],
                    rhs=[0],
                    names=['H_'+str(start_i[e])+'_'+str(start_j[e])])          


    for e in range(0,long):
        thevars=[]
        thecoefs=[]
        for k in range(int(position_ini[e]),int(position_fin[e])):
            if typ[k]==1 or typ[k]==3:
                thevars.append(v_var[k])
                thecoefs.append(1)
        thevars.append(z_var[int(start_i[e])])
        thecoefs.append(-1)
        cpx.linear_constraints.add(
                    lin_expr=[cplex.SparsePair(ind=thevars, val=thecoefs)],
                    senses=['L'],
                    rhs=[0],
                    names=['I_'+str(start_i[e])+'_'+str(start_j[e])])
        
    lg.debug('solving cplex problem!....')
    start_time = time.perf_counter()
    cpx.solve()  
    lg.debug(f'done solving cplex, took {time.perf_counter()-start_time}')

    l_names = cpx.linear_constraints.get_names()
    l_values = cpx.solution.get_dual_values()

    
    for constraint_name, constraint_value in zip(l_names, l_values):  
        if constraint_value == 0:
            continue
        c_source = c_dest = e_source = e_dest = None
    
        keys = constraint_name.split('_')
        constr_var = keys[0]
        if len(keys) > 1:
            c_source = keys[1]
            if len(keys) > 2:                
                c_dest = keys[2]
                if len(keys) > 4:
                    e_source = keys[3]
                    e_dest = keys[4]
    
        store_dual_value(res_dict,constr_var, constraint_value, c_source, c_dest, e_source, e_dest)

    return (res_dict, cpx.solution.get_objective_value())
