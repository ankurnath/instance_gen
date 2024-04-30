import cnfgen
import networkx as nx
import numpy as np
from pysat.solvers import Solver
import os
from scipy.sparse import csr_matrix,save_npz,load_npz
from multiprocessing.pool import Pool

def generate_hard_3_col_instances(n,seed,filename):

    np.random.seed(seed)
    G=nx.Graph()
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

    # print(G.edges())
    G=nx.to_numpy_array(G)
    sparse_matrix = csr_matrix(G)
    save_npz(filename, sparse_matrix)
    print('Saved file name:',filename)

def generate_instances(n,seeds,num_instances, folder_path, instance_type):
    arguments = []
    # seeds=
    for i in range(num_instances):
        # generate_hard_3_col_instances(n, os.path.join(folder_path, 
        #                                 f'3col_{n}vertices_{str(i).zfill(4)}.npz'))
        arguments.append((n,seeds[i], os.path.join(folder_path, f'3col_{n}vertices_{str(i).zfill(4)}.npz')))
    
    # with Pool() as pool:
    #     pool.starmap(generate_hard_3_col_instances, arguments,chunksize=4)
    
    step=os.cpu_count()
    # # step=4
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
    num_train=4000
    num_test=100
    num_val=50

    # num_train=4
    # num_test=4
    # num_val=10

    total_instance_per_distribution=num_train+num_test+num_val
    # distributions=[50]
    distributions=[50,100,150,200,300,400]

    # for n in [50,100,150,200,300,400]:
    _random_seeds=np.random.choice(np.arange(1e6,dtype=int),
                                  size=total_instance_per_distribution*len(distributions),
                                  replace=False).reshape(len(distributions),-1)
    
    for i,n in enumerate(distributions):
        distribution=f'3col_{n}vertices'
        random_seeds=_random_seeds[i]
        

        # random_seeds=np.random.choice(np.arange(1e6,dtype=int),size=num_train+num_test+num_val,
        #                               replace=False)

        os.makedirs(os.path.join(train_folder,distribution), exist_ok=True)
        generate_instances(n,random_seeds[:num_train],num_train, os.path.join(train_folder,distribution)
                           , "training")
        os.makedirs(os.path.join(test_folder,distribution), exist_ok=True)
        generate_instances(n,random_seeds[num_train:num_train+num_test],num_test, 
                           os.path.join(test_folder,distribution), "testing")
        # generate_instances(num_test,  os.path.join(test_folder,distribution), "testing")
        os.makedirs(os.path.join(val_folder,distribution), exist_ok=True)
        generate_instances(n,random_seeds[num_train+num_test:],num_val, 
                           os.path.join(val_folder,distribution), "validation")
        # generate_instances(num_val, os.path.join(val_folder,distribution), "validation")

        