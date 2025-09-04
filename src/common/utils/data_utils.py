import numpy as np
import pandas as pd
import csv,sys

from os import makedirs

__all__ = ['get_node_feature',
           'get_path_cost',
           'get_path_cost_2',
           'get_commodity_time',
           'get_commodity_demand',
           'write_results_from_paths'
           ]


def get_access_time(time_matrix, n, v):
    return np.sum(time_matrix)/(n*(n-1)) * v


# def get_time_and_demand_matrices(n_nodes: np.int32):
#     return (get_time_matrix(get_time_df(n_nodes), n_nodes), get_demand_matrix(get_demand_df(n_nodes), n_nodes))

# def get_time_df(n_nodes: np.int32) -> pd.DataFrame:
#     cost_file_name = f'../cabdata/cab{n_nodes}_c.csv'
#     #print(os.getcwd())
#     return (pd.read_csv(cost_file_name, header=0, dtype={'fromnode': np.int16, 'tonode': np.int16, 'c': np.int32}))


def get_node_feature(n: int, B: np.ndarray) -> np.ndarray:
    """
    returns the node features' vector

    Parameters
    ----------
    n : number of nodes
    B : commodities demand 2d matrix
    """
    node_feature = np.zeros(n)
    for i in range(0, n):
        for j in range(0, n):
            if i != j:
                node_feature[i] += B[i][j] + B[j][i]
        node_feature[i] = round(node_feature[i], 3)

    return node_feature

# def get_demand_df(n_nodes: np.int32) -> pd.DataFrame:
#     weight_file_name = f'../cabdata/cab{n_nodes}_w.csv'
#     return (pd.read_csv(weight_file_name, header=0, dtype={'fromnode': np.int16, 'tonode': np.int16, 'w': np.float32}))


def get_path_cost(i, j, path_time, node_features_list, A, r):
    """
    calculates and return the cost (profit) associated with the commodity
    using the improved time (path_time) found

    Parameters
    ----------
    i : commodity source node
    j : commodity target node
    path_time : accoumulated path travel time
    node_features_list : "promedio ponderado" vector
    A : time 2d matrix
    r : TODO complete this description

    """
    return round((A[i][j]-path_time)/(pow(path_time, r))*node_features_list[i]*node_features_list[j], 3)

def get_path_cost_2(info, source, destination, path_time):
    A = info.time_matrix
    node_features_list = info.node_feature_vector
    r = info.parameters.r
    
    return round((A[source][destination]-path_time)/(pow(path_time, r))*node_features_list[source]*node_features_list[destination], 3)


def get_direct_time_dict(t_m: np.ndarray):
    return {
        (i, j): val for i, row in enumerate(t_m) for j, val in enumerate(row)
    }


def random_choice_noreplace_vector(n, p, seed):
    np.random.seed(seed)
    return np.random.rand(n).argsort(-1)[:p]


def get_path_bin_sum_from_bin(path_binary):
    return np.sum([val*2**(i) for i, val in enumerate(path_binary)])


def get_path_bin(path, n):
    path_binary = np.zeros(n, dtype=bool)
    path_binary[path] = 1
    return path_binary

def get_path_bin_sum(path, n):
    '''
    path: hub set, i.e: [2,4,5] (p=3)
    n: number of nodes in the problem, i.e.: 10

    this method transforms the hub set into a binary list, i.e:
    [0, 0, 1, 0, 1, 1, 0, 0, 0, 0] --> (get_path_bin)
    then, gets and return the decimal number representation of the binary list:
    0 * 2^0 + 0 * 2^1 + 0 * 2^2 ...

    '''
    return get_path_bin_sum_from_bin(get_path_bin(path, n))

def get_selected_hub_nodes_binary_vector(hub_list: np.ndarray, node_count: int):
    z = np.repeat(0, node_count)
    z[hub_list] = 1
    return z

def get_selected_hub_edges_binary_matrix(hub_list: np.ndarray, node_count: int):
    v = np.zeros([node_count, node_count])
    for i, j in list(zip(hub_list[:-1], hub_list[1:])):
        v[i, j] = 1
        v[j, i] = 1
    return v

def get_edge_time(edge, time_matrix):
    s, t = edge
    return time_matrix[s, t]

def get_hub_node_time(edge, alpha, time_matrix):
    return get_edge_time(edge, time_matrix) * alpha

def remove_matrix_diagonal(matrix: np.ndarray) -> np.ndarray:
    mask = np.ones(matrix.shape, dtype=bool)
    mask[np.diag_indices(matrix.shape[0])] = False
    return matrix[mask].reshape(matrix.shape[0], -1)

def get_total_profit(or_time_mat, new_time_mat, feature_vec, r) -> np.float32:
    or_time_mat_p = remove_matrix_diagonal(or_time_mat)
    new_time_mat_p = remove_matrix_diagonal(new_time_mat)
    feature_vec_p = remove_matrix_diagonal(
        feature_vec * feature_vec.reshape(feature_vec.shape[0], 1))
    return np.sum(((or_time_mat_p - new_time_mat_p) / (new_time_mat_p ** r)) * feature_vec_p)

def get_path_type(path: np.ndarray, v: np.ndarray) -> int:
    """
    returns the path type based on the following rules:\n
    1 -> all hub edges\n
    2 -> access edge + hub edges\n
    3 -> hub edges + exit edge\n
    4 -> access edge + hub edges + exit edges

    Parameters
    ----------
    path : the path (vector) being analyzed 
    v : the binary hub archs 2d matrix
    """
    has_access_edge = not bool(v[path[0], path[1]])
    has_exit_edge = not bool(v[path[-2], path[-1]])
    type = 1
    if has_access_edge:
        if has_exit_edge:
            type = 4
        else:
            type = 2
    else:
        if has_exit_edge:
            type = 3

    return type

def get_results_from_paths(general_params, general_info, paths, hubs):
    return [
        (
            path[0],  # path
            path[0][0],  # start
            path[0][-1],  # end
            get_path_type(path[0],get_selected_hub_edges_binary_matrix(hubs, general_params.n)),  # type
            round(path[1], 3),  # time
            get_path_cost(
                paths[path][0][0],
                paths[path][0][-1],
                paths[path][1],
                general_info.node_feature_vector,
                general_info.time_matrix,
                general_params.r)
        ) for path in paths]


def write_results_from_paths(general_params, results: list, reported_time, suffix=''):
    makedirs('../paths', exist_ok=True)
    file_name = f'../paths/data_{general_params.n}_{general_params.p}_{general_params.alpha}_{general_params.perc}_0.1_{general_params.r}_3{suffix}.csv'
    all_edges_no_diag = [(i, j) for i in range(general_params.n) for j in range(general_params.n) if i != j]
    with open(file_name, 'w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        # writer.writerow([round(finish-start,4)])
        writer.writerow([reported_time])
        writer.writerow(all_edges_no_diag)
        writer.writerows(results)


def get_hubs(path, type=1):
    if len(path) == 0:
        return []
    cases = {
        1: path,
        2: path[1:],
        3: path[:-1],
        4: path[1:-1]
    }
    return cases.get(type, None)

def get_reverse_path(path:list):
    rev_path = path.copy()
    rev_path.reverse()
    return rev_path
    
def ifnull(pos, val):
    if len(sys.argv) <= pos or (sys.argv[pos] == None):
        return val
    return sys.argv[pos]

def clean_concat_paths_df(paths_df1, paths_df2):
  res_df = pd.concat([paths_df1,paths_df2], ignore_index=True)
  res_df['path'] = res_df['path'].apply(tuple)
  res_df.drop_duplicates(subset=['path','type'], keep='first',ignore_index=True, inplace=True)
  res_df['path'] = res_df['path'].apply(list)
  return res_df
