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

from torch.utils.data import DataLoader, TensorDataset

import sensors.utils.utils as utils
import sensors.utils.fault_detection as fd
import sensors.nn.models as models

from pyprojroot import here
root_dir = str(here())

data_dir = '~/data/interim/'

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

warnings.simplefilter("ignore")

def train_AE(model, X, N_epochs, batch_size, lr, training_loss):
 
    dataset = TensorDataset(X, X)  # we want to reconstruct the same input
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    model.reset_parameters()

    for epoch in range(N_epochs):
        for batch, _ in dataloader:
            
            optimizer.zero_grad()
            output = model(batch)
            loss = training_loss(batch, output)
            loss.backward()
            optimizer.step()

    return model

def evaluate_model(model, data, labels, N_epochs, batch_size=2048, lr=1e-3, training_loss=torch.nn.MSELoss()):

    auc_list = []
    for i in range(data.shape[0]):

        X = data[i,:,:]
        label = labels[i,:]
        model = train_AE(model, X, N_epochs, batch_size, lr, training_loss)

        model.eval()
        Y = model(X)

        score_function = torch.nn.MSELoss(reduction='none')
        score = torch.mean(score_function(X,Y), axis=1).cpu().detach().numpy()

        tpr, fpr, _ = utils.roc_params(metric=score, label=label, interp=True)
        auc = utils.compute_auc(tpr,fpr)

        auc_list.append(auc)
    
    return auc_list

def main(args):

    if args.device=='auto':
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
    data = torch.tensor(data).float()
    data = data.to(device)
    
    exp_template = f'{args.anomaly_partition}parts_{args.anomaly_level}level_{args.n_anomalies}anomaly'

    n_timestamps = data.shape[2]

    def objective(trial):

        gc.collect()

        # Possible hyperparameters     
        N_epochs = trial.suggest_categorical('N_epochs', args.N_epochs)
        n_layers = trial.suggest_categorical('n_layers', args.n_layers)
        reduction = trial.suggest_categorical('reduction', args.reduction)
        batch_size = trial.suggest_categorical('batch_size', args.batch_size)
        lr = trial.suggest_categorical('lr', args.lr)

        training_loss = torch.nn.MSELoss()

        ###

        print(f"Trial: {trial.number}", flush=True)
        print(f"- N Epochs: {N_epochs}", flush=True)
        print(f"- N Layers: {n_layers}", flush=True)
        print(f"- Reduction: {reduction}", flush=True)
        print(f"- Batch size: {batch_size}", flush=True)
        print(f"- Learing rate: {lr}", flush=True)
        print(f"- Training loss: {training_loss}", flush=True)

        ###

        model = models.AE(n_timestamps, n_layers, reduction)                        
        model = model.to(device)

        if model.encoder[-2].out_features < 2:
            raise optuna.TrialPruned()

        auc_list = evaluate_model(model, data, labels, N_epochs, batch_size, lr, training_loss)
        
        trial.set_user_attr("std_auc", np.std(auc_list))
        trial.set_user_attr("min_auc", np.min(auc_list)) 
        trial.set_user_attr("auc_list", [round(elem, 2) for elem in auc_list])

        return np.mean(auc_list).round(3)


    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir,exist_ok=True)

    study = optuna.create_study(sampler=TPESampler(), direction='maximize',
                                study_name='maximize_autoencoder_auc: ' + exp_template,
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=5,
                                                                   n_warmup_steps=24,
                                                                   interval_steps=6))
    
    study.set_metric_names(['auc'])

    log_file = args.log_dir + 'log_' + exp_template + args.log_mod + '.pkl'

    if args.reuse:
        if os.path.isfile(log_file):
            print('Reusing previous study', flush=True)
            study = joblib.load(log_file)

    study.optimize(objective, n_trials=args.n_trials, gc_after_trial=True)
    joblib.dump(study, log_file)

    print('____ END OF STUDY ___\n\n')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--n_trials', type=int, default=1)
    parser.add_argument('--log-dir', type=str, default=root_dir+'/outputs/AE_optuna/')
    parser.add_argument('--log_mod', type=str, default='')

    parser.add_argument('--data_size', type=int, default=100)
    parser.add_argument('--anomaly_level', type=float, default=10)
    parser.add_argument('--anomaly_partition', type=int, default=20)
    parser.add_argument('--n_anomalies', type=int, default=1)

    parser.add_argument('--N_epochs', type=int, nargs='+', default=[50, 75, 100, 125, 150, 200, 500])
    parser.add_argument('--n_layers', type=int, nargs='+', default=[1, 2, 3, 4, 5])
    parser.add_argument('--reduction', type=float, nargs='+', default=[0.25, 0.4, 0.5, 0.6, 0.75])
    parser.add_argument('--batch_size', type=int, nargs='+', default=[2048])
    parser.add_argument('--lr', type=float, nargs='+', default=[1e-2, 1e-3, 1e-4])

    parser.add_argument('--device', type=str, default='cuda')

    parser.add_argument('--reuse', action='store_true', default=True)
    parser.add_argument('--no-reuse', dest='reuse', action='store_false')

    args = parser.parse_args()

    main(args)

