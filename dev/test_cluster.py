import numpy as np
import pandas as pd

import os
import gc
import argparse
import torch
import optuna
import joblib
import warnings

from optuna.samplers import TPESampler
from sklearn.cluster import KMeans
from tqdm import tqdm

import sensors.utils.utils as utils
import sensors.utils.fault_detection as fd
import sensors.nn.clustering as cl

from pyprojroot import here
root_dir = str(here())

data_dir = '~/data/interim/'

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

warnings.simplefilter("ignore")

def train_cluster(N_epochs, model, X, G, device, weight_loss=0.25, lr=1e-3):
 
    loss_evo = []
    loss_mc_evo = []
    loss_o_evo = []

    X = torch.tensor(X)
    X = X.to(device)

    # Node coordinates
    C = torch.tensor(G.coords)
    A = torch.tensor(G.W.toarray()).float() #Using W as a float() tensor
    A = A.to(device)    

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    num_clusters_per_feature = [5, 6]
    kmeans_feats = cl.kmeans_features(C, num_clusters_per_feature).to(device).float()
    n_extra_feats = kmeans_feats.shape[1]

    model.train()
    model.reset_parameters()
    # for epoch in tqdm(range(N_epochs)):
    for epoch in range(N_epochs):
        optimizer.zero_grad()
        S, loss_mc, loss_o = model(X, A, kmeans_feats)
        loss = loss_mc + weight_loss*loss_o
        loss.backward()
        optimizer.step()
        loss_evo.append(loss.item())
        loss_mc_evo.append(loss_mc.item())
        loss_o_evo.append(loss_o.item())

    return S

def evaluate_model(model, weight_loss, N_epochs, data, labels, data_dfs, G, nodes_orig, device):

    cluster_score_list = []
    auc_list = []
    for i in range(data.shape[0]):
        X = data[i,:,:]
        label = labels[i,:]
        df_anomaly = data_dfs[i]

        nodes = nodes_orig.copy()

        S = train_cluster(N_epochs, model, X, G, device, weight_loss)

        cluster_score, auc = utils.get_score(nodes, df_anomaly, S)
        
        cluster_score_list.append(cluster_score)
        auc_list.append(auc)
    
    return cluster_score_list, auc_list

def main(args):

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = args.device

    print(f'Using {device}', flush=True)
    print(f'log dir: {args.log_dir}', flush=True)
    print(f'------', flush=True)

    rng_seed = 0
    torch.manual_seed(rng_seed)
    torch.cuda.manual_seed(rng_seed)
    np.random.seed(rng_seed)

    
    # OBTAINING DATA
    dataset = 'df_StOlavs_D1L2B'
    df_orig = pd.read_parquet(data_dir + f'{dataset}.parq')

    df, nodes = fd.treat_nodes(df_orig)
    _, nodes['subgraph'] = fd.NNGraph(nodes, radius=15, subgraphs=True)

    main_graph = nodes.subgraph.value_counts().index[0]
    nodes = nodes.query('subgraph==@main_graph').copy()
    G = fd.NNGraph(nodes, radius=15)
    df = df[df.pid.isin(nodes.pid.unique())].copy()   

    data, labels, data_dfs = utils.generate_cluster_anomaly(df, nodes, G,
                                                            data_size=args.data_size,
                                                            partition=args.anomaly_partition,
                                                            anomaly_level=args.anomaly_level,
                                                            n_anomalies=args.n_anomalies)
    
    exp_template = f'{args.anomaly_partition}parts_{args.anomaly_level}level_{args.n_anomalies}anomaly'

    n_timestamps = data.shape[2]

    def objective(trial):

        gc.collect()

        # Possible hyperparameters
        n_extra_feats = 2
        conv1d_n_feats = 3
        conv1d_kernel_size = 60
        conv1d_stride = 30
        graphconv_n_feats = 30

        N_epochs = trial.suggest_categorical('N_epochs', args.N_epochs)
        n_clusters = trial.suggest_categorical('n_clusters', args.n_clusters)
        weight_loss = trial.suggest_categorical('weight_loss', args.weight_loss)
        weight_coords = trial.suggest_categorical('weight_coords', args.weight_coords)

        ###

        print(f"Trial: {trial.number}", flush=True)
        print(f"- N Epochs: {N_epochs}", flush=True)
        print(f"- N Clusters: {n_clusters}", flush=True)
        print(f"- Weight (loss): {weight_loss}", flush=True)
        print(f"- Weight (coords): {weight_coords}", flush=True)

        ###

        model = cl.ClusterTS(conv1d_n_feats, conv1d_kernel_size, conv1d_stride, graphconv_n_feats,
                        n_timestamps, n_clusters, n_extra_feats, weight_coords)
        model = model.to(device)

        cluster_score_list, auc_list = evaluate_model(model, weight_loss, N_epochs,
                                                      data, labels, data_dfs, G, nodes, device)

        trial.set_user_attr("mean_auc", np.mean(auc_list))
        trial.set_user_attr("std_auc", np.std(auc_list))
        trial.set_user_attr("min_cscore", np.min(cluster_score_list)) 
        trial.set_user_attr("std_cscore", np.std(cluster_score_list))
        trial.set_user_attr("cscore_list", [round(elem, 2) for elem in cluster_score_list])
        trial.set_user_attr("auc_list", [round(elem, 2) for elem in auc_list])

        return np.mean(cluster_score_list).round(3)


    study = optuna.create_study(sampler=TPESampler(), direction='maximize',
                                study_name='maximize_cluster_score: ' + exp_template,
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=5,
                                                                   n_warmup_steps=24,
                                                                   interval_steps=6))
    
    study.set_metric_names(['cscore'])

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir,exist_ok=True)

    log_file = args.log_dir + 'log_' + exp_template + '.pkl'

    if args.reuse:
        if os.path.isfile(log_file):
            study = joblib.load(log_file)

    study.optimize(objective, n_trials=args.n_trials, gc_after_trial=True)
    joblib.dump(study, log_file)

    print('____ END OF STUDY ___\n\n')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--n_trials', type=int, default=1)
    parser.add_argument('--log-dir', type=str, default=root_dir+'/outputs/')

    parser.add_argument('--data_size', type=int, default=100)
    parser.add_argument('--anomaly_level', type=float, default=10)
    parser.add_argument('--anomaly_partition', type=int, default=20)
    parser.add_argument('--n_anomalies', type=int, default=1)

    parser.add_argument('--N_epochs', type=int, nargs='+', default=[500, 1000, 5000, 1000])
    parser.add_argument('--n_clusters', type=int, nargs='+', default=[5, 10, 15, 20])
    parser.add_argument('--weight_loss', type=float, nargs='+', default=[0, 0.25, 0.5, 0.75, 1])
    parser.add_argument('--weight_coords', type=float, nargs='+', default=[0, 0.25, 0.5, 0.75, 1])

    parser.add_argument('--device', type=str, default='cuda')

    parser.add_argument('--reuse', action='store_true', default=True)
    parser.add_argument('--no-reuse', dest='reuse', action='store_false')

    args = parser.parse_args()

    main(args)

