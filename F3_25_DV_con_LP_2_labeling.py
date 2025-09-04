import time
import cplex
import numpy as np
import pandas as pd
from os.path import exists
from os import makedirs
import csv
from F_utils import get_hubs, ifnull, get_current_datetime, removeDuplicates, read_cabdata

init_model_time=time.time()
n = int(ifnull(1, 20))                      # Number of nodes
alpha = float(ifnull(2, 0.2))               # Discount factor
v = float(ifnull(3, 0.1))                   # Access time factor
p = int(ifnull(4,7))                        # Number of hubs
perc = float(ifnull(5,1))                   # Percentage of edges used 
ex= int(ifnull(6,1))                        # experiment id (unused)
r=float(ifnull(7,1.7))                      # r function param
seed = ifnull(8,None)                       # random seed (unused)
process_heuristic_paths = eval(ifnull(9,'False'))   # heuristic parameter process flag
process_localsearch_paths = eval(ifnull(10,'False'))   # local search parameter process flag
problem_type = ifnull(11,'CABDATA')

total_start_time = time.time()


A = list()
B = list()
data_dir=''
if problem_type == 'MTL':
    A, B = read_cabdata(n,'_mtl')
    data_dir = 'mtl'
else:
    A, B = read_cabdata(n,'')
    data_dir = 'cabdata'

read_file = ''
print(f'process_localsearch_paths: {process_localsearch_paths}, parametro: {ifnull(10,False)}, tipo:{type(ifnull(10,False))}')
if process_localsearch_paths:
    read_file=f'./paths/localsearch/{data_dir}/data_{n}_{p}_{alpha}_{perc}_{v}_{r}_4_localsearch.csv'
else:
    # data de CG
    read_file=f'./paths/cg/{data_dir}/data_{n}_{p}_{alpha}_{perc}_{v}_{r}_4_setting.csv'
    
edges = []
path_generation_time = 0

with open(read_file, newline='') as f:
    reader = csv.reader(f, delimiter='\t')
    for idx, row in enumerate(reader):
        if idx==0: #primera fila es el tiempo
            path_generation_time = row[0]
        elif idx==1: # segunda fila son los edges
            [edges.append(eval(item)) for item in row];
        else: # de aqui en adelante cada linea tiene path, 
            break

paths_df = pd.read_csv(
    read_file, 
    sep='\t', 
    header=None, 
    skip_blank_lines=True, 
    skiprows=2, 
    names=['path','start','end','type','distance','cost'])

init_prepro_time=time.time()
paths_df['path'] = paths_df['path'].apply(lambda path: eval(path))
paths_df['hubs'] = paths_df[['path','type']].apply(lambda r: get_hubs(r.path, r.type),axis=1)


# #---------- Index i, index j, index e----------------------#
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
number_of_edges=len(edges)
#aux1=len(paths_df)
aux1=paths_df['path'].count()
path_time=eval(path_generation_time)
init = paths_df['start'].to_numpy()
end = paths_df['end'].to_numpy()
typ = paths_df['type'].to_numpy()
cost = paths_df['cost'].to_numpy()
dist = paths_df['distance'].to_numpy()


node_features_list=np.zeros(n)
for i in range(0,n):
  for j in range(0,n):
    if i!=j:
      node_features_list[i]+=B[i][j]+B[j][i]
  node_features_list[i]=round(node_features_list[i],3)


##------------------------------------------------------------------------##
tst1 = pd.Series(zip(init,end))
df=paths_df.groupby(['start','end']).size().reset_index(name='Sizes')
long=len(df) 
auxiliar=df['Sizes'].to_numpy() 
position_fin=auxiliar.cumsum() 
position_ini=np.append(0, position_fin[:-1])


##------------------------------------------------------------------------##
used_edges=tst1.drop_duplicates()
start_i, start_j = zip(*used_edges)

H=np.zeros((aux1,nedges))
hubs = paths_df['hubs'].to_numpy()
for idx, hub in enumerate(hubs):
    for h_ini, h_fin in zip(hub[:-1],hub[1:]):
        H[idx, index_e[h_ini, h_fin]]=1
end_prepro_time=time.time()
t_prepro=end_prepro_time-init_prepro_time

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

## -------------------------------------  -----------------------------------------
end_prepro_time=time.time()
t_prepro=end_prepro_time-init_prepro_time
print("t_prepro ",t_prepro)
print("Solving the model\n")
counting=0 
cpx = cplex.Cplex()
cpx.parameters.timelimit.set(86400)
cpx.parameters.threads.set(4)

cpx.objective.set_sense(cpx.objective.sense.maximize)

v_var=list(
    cpx.variables.add(
        obj=cost2,
        lb=[0.0]*aux1,
        ub=[1]*aux1,
        names=['v' + str(k) for k in range(aux1)]
    ))
y_var=[cpx.variables.add(obj=[0 for j in range(0,n)],lb=[0.0]*n,ub=[1]*n,names=['y_(%d)(%d)' % (i, j) for j in range(0,n)]) for i in range(0,n)]

sume=np.zeros(nedges)

for e in range(0, nedges):
    sume[e]=np.sum(H[0:aux1,e:e+1])

for e in range(0,nedges):
    if sume[e]<0.5:
        cpx.variables.set_upper_bounds(y_var[int(index_i[e])][int(index_j[e])],0)
        cpx.variables.set_upper_bounds(y_var[int(index_j[e])][int(index_i[e])],0)
for i in range(0,n):
    cpx.variables.set_upper_bounds(y_var[i][i],0)
       
z_var=list(cpx.variables.add(obj=[0]*n,lb=[0.0]*n,ub=[1]*n,names=['z'+str(i) for i in range(0,n)]))

l_var=list(
    cpx.variables.add(
        obj=[0]*n,
        lb=[0]*n,
        ub=[cplex.infinity]*n,
        names=['l_'+str(i) for i in range(0,n)]
    ))

#------------------------------------------Constraints--------------

#-----------------------------------------------# 
# Constraint  : ensure that p hubs are opened
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
        names=['pHubs'])

# Constrainst : establish that p-1 hub edges must be opened 
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
        names=['HubLines'])

# Constraints : is to obtain a line oriented to the node
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
        names=['open'+str(j)])
    

#---------------------------------
# Miller-Tucker Zemlin (MTZ) 
# Constraint: is to take advantage of providing
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
                names=['MTZ_'+str(i)+'.'+str(j)])

# Constraint : restrict that each commodity is transported either by respectively,
# using the hub line or taking direct path
#--------- sum_{k\in r}v_k+e_r=1  -----------------#  
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
                names=['Paths_'+str(int(start_i[e]))+'.'+str(int(start_j[e]))])
    
##--------- 
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
                    names=['AllowedPaths_'+str(int(start_i[ij]))+'.'+str(int(start_j[ij]))+'.'+str(index_i[e])+'.'+str(index_j[e])])
    
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
        names=['new1_'+str(i)+','+str(k)])
        
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
                names=['Paths1_'+str(start_i[e])+'.'+str(start_j[e])])          
        
#    #-------------------------------------------------------------------------##
    
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
                names=['Paths3_'+str(start_i[e])+'.'+str(start_j[e])])
                

#cpx.parameters.mip.strategy.search.set(cpx.parameters.mip.strategy.search.values.traditional)#
#cpx.parameters.preprocessing.presolve.set(cpx.parameters.preprocessing.presolve.values.off)
#cpx.parameters.threads.set(1)
#lazy=cpx.register_callback(MyLazyConsCallback)
#lazy.counting = counting   
cpx.write("prob.lp")
#lazy=cpx.register_callback(MyLazyConsCallback)
#counting=lazy.counting
end_model_time=time.time()  
#print("tiempo carga modelo ",end_model_time-init_model_time)

init_solve_time0=time.time()  
cpx.solve()  
end_solve_time0=time.time()

status_lp=cpx.solution.get_status()
solution_lp=cpx.solution.get_objective_value()

cpx.variables.set_types([(v_var[i], cpx.variables.type.binary) for i in range(0,aux1)]) 
cpx.variables.set_types([(y_var[i][j], cpx.variables.type.binary) for i in range(0,n) for j in range(0,n)]) 
cpx.variables.set_types([(z_var[i], cpx.variables.type.binary) for i in range(0,n)])  

init_solve_time=time.time()    
cpx.solve()    
end_solve_time=time.time()

model_time=end_model_time-init_model_time
solve_time=end_solve_time-init_solve_time
total_time=path_time+model_time+solve_time
solve_relaxed_time=end_solve_time0-init_solve_time0

nousa=0 
contador=0;
paths_v = []
for k in range(0,aux1):
    if cpx.solution.get_values(v_var[k])>0.5:
        paths_v.append(k)
        contador=+1
using_structure=contador/(n*(n-1))*100
## --------------------------------------------------

func='f3_25_DV_con_lp'
rel_mip_gap = cpx.solution.MIP.get_mip_relative_gap()
abs_mip_gap = cpx.solution.MIP.get_best_objective()

hub_nodes,hub_edges = [],[]
for i in range(0,n):
    if cpx.solution.get_values(z_var[i])>0.5:
        hub_nodes.append(i)
for i in range(0,n):
    for j in range(0,n):
        if cpx.solution.get_values(y_var[i][j])>0.5 and (i,j) not in hub_edges:
            hub_edges.append((i,j))

tmp = []
for k in range(0,aux1):
    if (cpx.solution.get_values(v_var[k])>0.5):
        tmp.append(k)

path_list = paths_df.loc[tmp, ["path","type"]].to_numpy()
acc_edges_tmp = []
ex_edges_tmp = []
for path, t in path_list:
    if t in(3,4):
        ex_edges_tmp.append(path[-2:])
    if t in(2,4):
        acc_edges_tmp.append(path[0:2])

access_edges = removeDuplicates(acc_edges_tmp)
exit_edges = removeDuplicates(ex_edges_tmp)

#cpx.write('./results/mymodel.lp') 
result1, result2 = '',''

results_path='./results'
if process_localsearch_paths:
    results_path += f'/localsearch/{data_dir}'
else: 
    results_path += f'/cg/{data_dir}'

makedirs(results_path, exist_ok=True)

if process_heuristic_paths:
    result1=f'{results_path}/summary_heuristic.csv'
    result2=f'{results_path}/details_heuristic.csv'
else:
    result1=f'{results_path}/summary.csv'
    result2=f'{results_path}/details.csv'

f1_mode='w'
f2_mode='w'

if (exists(result1)):
    f1_mode='a'

if (exists(result2)):
    f2_mode='a'

with open(result1, f1_mode, newline='') as f:
    writer = csv.writer(f, delimiter=';')
    #
    if f1_mode != 'a':
        writer.writerow(['model','n','p','v','r','alpha','perc','ex','n_edges',
        'n_paths','path_time','t_prepro','model_time','solv_time','total_time','UB','LB',
        'using_structure','counting','status','status_lp','solve_relaxed_time','solution_lp'])
    # 
    writer.writerow([
        func,
        n,
        p,
        v,
        r,
        alpha,
        perc,
        seed,
        number_of_edges,
        aux1,
        round(path_time,3),
        round(t_prepro,3),
        round(model_time,3),
        round(solve_time,3),
        round(total_time,3),
        round(cpx.solution.MIP.get_best_objective(),3),
        round(cpx.solution.get_objective_value(),3),
        round(using_structure,3),
        counting,
        cpx.solution.get_status(),
        status_lp,
        round(solve_relaxed_time,3),
        round(solution_lp,3)
    ])

with open(result2, f2_mode, newline='') as f:
    writer = csv.writer(f, delimiter=';')
    if f1_mode != 'a':
        writer.writerow(['fecha','formula','n','p','alpha','perc','seed','v','r','total time','rel mip gap','abs mip gap','hub nodes','hub edges', 'access edges','exit edges','paths_sel'])#, 'direct edges'])
    writer.writerow([get_current_datetime(),func,n,p,alpha,perc,seed,v,r, total_time, rel_mip_gap, abs_mip_gap, hub_nodes,hub_edges,access_edges, exit_edges, paths_v])#, direct_edges])
