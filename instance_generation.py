import cnfgen
import networkx as nx
import numpy as np
from pysat.solvers import Solver
import os
from scipy.sparse import csr_matrix,save_npz,load_npz
from multiprocessing.pool import Pool

def generate_hard_3_col_instances(n,filename):

    G=nx.Graph()
    u,v = np.random.randint(0, n, 2)
    satisfiable=True
    while satisfiable:
        u,v = np.random.randint(0, n, 2)
        if u==v:
            continue
        else:
            G.add_edge(u, v)
            formula=cnfgen.GraphColoringFormula(G,colors=3)
            s = Solver(bootstrap_with=formula.clauses())
            satisfiable=s.solve()
            if not satisfiable:
                G.remove_edge(u,v)

    G=nx.to_numpy_array(G)
    sparse_matrix = csr_matrix(G)
    save_npz(filename, sparse_matrix)
    print('Saved file name:',filename)

def generate_instances(num_instances, folder_path, instance_type):
    arguments = []
    for i in range(num_instances):
        # generate_hard_3_col_instances(n, os.path.join(folder_path, 
        #                                 f'3col_{n}vertices_{str(i).zfill(4)}.npz'))
        arguments.append((n, os.path.join(folder_path, f'3col_{n}vertices_{str(i).zfill(4)}.npz')))
    
    # with Pool() as pool:
    #     pool.starmap(generate_hard_3_col_instances, arguments,chunksize=4)
    
    step=os.cpu_count()
    step=4
    for i in range(0,num_instances,step):
    # print(f'Starting generating {instance_type} instances')
        with Pool(step) as pool:
            # pool.starmap_async(generate_hard_3_col_instances, arguments).wait()
            pool.starmap(generate_hard_3_col_instances, arguments[i:i+step])
    print(f'Finished generating {instance_type} instances')


if __name__ == '__main__':
    save_folder='../data_color'

    train_folder=os.path.join(save_folder,'training')
    test_folder=os.path.join(save_folder,'testing')
    val_folder=os.path.join(save_folder,'validation')

    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)
    # for n in [50,100,150,200,300,400]:
    for n in [400]:
        distribution=f'3col_{n}vertices'
        
        num_train=10
        num_test=10
        num_val=10

        os.makedirs(os.path.join(train_folder,distribution), exist_ok=True)
        generate_instances(num_train, os.path.join(train_folder,distribution), "training")
        os.makedirs(os.path.join(test_folder,distribution), exist_ok=True)
        generate_instances(num_test,  os.path.join(test_folder,distribution), "testing")
        os.makedirs(os.path.join(val_folder,distribution), exist_ok=True)
        generate_instances(num_val, os.path.join(val_folder,distribution), "validation")

        