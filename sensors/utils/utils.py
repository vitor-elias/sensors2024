import numpy as np
import geopandas as gpd
import plotly.express as px

import torch

from sklearn.metrics import mean_squared_error, confusion_matrix, auc, f1_score
from sklearn.cluster import KMeans

import sensors.utils.fault_detection as fd

def visualize_map(nodes_input, color='smoothed', size=None, size_max = None, animation_frame=None, hover_data=[None],
                  colormap = px.colors.diverging.oxy, zoom=15, range_color = None, opacity=1,
                  title=None, transparent=False, discrete_colormap = px.colors.qualitative.Light24,
                  figsize=(1200,800), renderer=None, mapbox_style='carto-positron', savehtml=None):
    
    columns = [col for col in ['pid', 'easting','northing',color, animation_frame]+hover_data if col is not None]
    columns = list(set(columns))

    if isinstance(size,str):
        columns.append(size)

    nodes = nodes_input[columns].copy()

    nodes_gdf = gpd.GeoDataFrame(nodes, geometry=gpd.points_from_xy(nodes.easting,nodes.northing), crs='3035')
    nodes_gdf = nodes_gdf.to_crs('4326')

    if animation_frame is not None:
        nodes_gdf[animation_frame] = nodes_gdf[animation_frame].astype(str)
        nodes_gdf = nodes_gdf.sort_values([animation_frame, 'pid'])

    if range_color is None:
        range_color = (nodes_gdf[color].min(), nodes_gdf[color].max())
        
    fig = px.scatter_mapbox(nodes_gdf, lat=nodes_gdf.geometry.y, lon=nodes_gdf.geometry.x,
                            hover_name = 'pid', hover_data = hover_data, opacity=opacity,
                            color=color, size=size, size_max = size_max,
                            mapbox_style=mapbox_style, animation_frame=animation_frame,
                            width=figsize[0], height=figsize[1], zoom=zoom, color_discrete_sequence=discrete_colormap,
                            color_continuous_scale=colormap, range_color=range_color,
                            )
    

    # cbar_y = 0.775 if animation_frame is not None else 0.9
    fig.update_layout(coloraxis={'colorbar': {'title': {'text': ''},
                                            'len':0.5,
                                            # 'y':cbar_y,
                                            'thickness':5
                                            }})
    fig.update_layout(title=title)
    if transparent:
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")

    if savehtml is not None:
        fig.write_html(savehtml)

    fig.show(renderer=renderer)

def roc_params(metric, label, interp=True):
    fpr = []
    tpr = []
    thr = []
    thr_list = list(np.linspace(0, metric.max(),1001))

    fp = 1
    ind = 0
    while fp > 0:
        threshold = thr_list[ind]
        ind += 1

        y = (metric>threshold)
        tn, fp, fn, tp = confusion_matrix(label, y).ravel()

        fpr.append( fp/(tn + fp) )
        tpr.append( tp/(tp + fn) )
        thr.append( threshold )

    while tp > 0:
        threshold = thr_list[ind]
        ind += 1
        y = (metric>threshold)
        tn, fp, fn, tp = confusion_matrix(label, y).ravel()

    
    fpr = fpr[::-1]
    tpr = tpr[::-1]
    thr = thr[::-1]

    if interp:
        fpr_base = np.linspace(0, 1, 101)
        tpr = list(np.interp(fpr_base, fpr, tpr))
        thr = list(np.interp(fpr_base, fpr, thr))
        fpr = list(fpr_base)

    fpr.insert(0, 0)
    tpr.insert(0, 0)
    thr.insert(0, threshold)

    return tpr, fpr, thr

def compute_auc(tpr, fpr):
    auc = 0
    for i in range(1, len(fpr)):
        auc += (fpr[i] - fpr[i - 1]) * (tpr[i] + tpr[i - 1]) / 2
    return auc

def generate_synthetic_graph(N, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    N_grid_points = int(np.floor(0.9*N))
    N_rand_points = int(np.ceil(0.1*N))

    connected = False

    while not connected:

        # Generating grid points

        # Initialize starting position
        current_position = [0, 0]

        # Store visited points
        visited_points = set()
        visited_points.add(tuple(current_position))

        # Generate N points
        while len(visited_points) < N_grid_points:
            # Update the current position by taking a vertical or horizontal step
            # horizontal can be +-5, vertical can be +-15

            if np.random.rand()>0.5:
                current_position[0] += np.random.choice([-5, 5])
            else:
                current_position[1] += np.random.choice([-14, 14])

            # Add the new position to the set of visited points
            visited_points.add(tuple(current_position))

        points_list = list(visited_points)

        # Generating random points
        reference_pos = np.random.choice(a=N_grid_points, size=N_rand_points, replace=False)
        reference_points = [list(points_list[i]) for i in reference_pos]

        for point in reference_points:
            point[0] += 10*np.random.random()
            point[1] += 10*np.random.random()
            visited_points.add(tuple(point))
        

        # Convert the set of visited points to a list
        pos = np.array(list(visited_points))
        pos = pos + 0.5*np.random.randn(pos.shape[0], pos.shape[1])

        radius=15
        sigma=radius**2

        G = pygsp.graphs.NNGraph(pos,
                                NNtype='radius',
                                epsilon = radius, sigma = sigma,
                                center=False, rescale=False)
        connected = G.is_connected()

    return G


def generate_ramp(G, size):
    # Auxiliary function for generating data.
    # Returns array with "size" random ramps on the given graph. A ramp emulates regions with different behaviors.
    # Direction of each ramp is randomly horizontal or vertical
    # Slope, start and end positions of the ramp are random, within some constraints.
    
    pos = G.coords
    min_slope = 0.1
    max_slope = 1
    
    # Initialize an empty matrix to store ramp vectors
    ramp_matrix = np.zeros((len(pos), size))

    for i in range(size):
        # Randomly choose the direction of the ramp (0, horizontal or 1, vertical)
        if np.random.rand() > 0.5:
            direction = 0
        else:
            direction = 1

        # Generate a random slope within the specified maximum slope
        slope = np.random.uniform(low=min_slope, high=max_slope)

        # Calculate the minimum ramp length as a fraction of the peak-to-peak range
        min_slope_dist = 0.25 * pos[:, direction].ptp()

        # Determine the starting and ending positions
        start = pos[:, direction].min() + pos[:, direction].ptp() * np.random.rand() * 0.5
        end = start + min_slope_dist + (pos[:, direction].max() - start - min_slope_dist) * np.random.rand()

        # Generate the ramp vector based on the slope and the position within the range
        ramp = slope * (pos[:, direction] - start)

        # Applying a mask that defines where the ramp exists
        mask = (pos[:, direction] >= start) * (pos[:, direction] < end)
        ramp = ramp * mask

        # Set values beyond the end of the ramp to the maximum ramp value
        ramp[pos[:, direction] >= end] = ramp.max()

        # Store the ramp vector in the matrix
        ramp_matrix[:, i] = ramp

    return ramp_matrix


def generate_smooth(G, size):
    # Auxiliary function for generating data
    # Returns array with "size" smooth signals by filtering white noise with the two first eigenvectors of the graph
    
    w, V = np.linalg.eigh(G.L.toarray())

    # Low-pass filter
    h = np.ones(len(w))
    h[0] = 1
    h[1] = 0.1
    h[2:] = 0 

    # Generating and filtering white noise to create a smooth graph signal
    displacement = np.random.randn(G.N, size) 
    displacement = V @ np.diag(h) @ V.T @ displacement

    # Normalizing signal: average power sum(x^2)/N = 1
    displacement = np.sqrt(G.N)*displacement/(np.linalg.norm(displacement,axis=0))

    return displacement


def pick_non_adjacent(G, n):
    # Auxiliary function for generating anomalies.
    # Picks n nodes that are not adjacent to each other.
    A = G.A.toarray()
    np.fill_diagonal(A,1)

    tries = 0

    while tries < 100000:
        tries+=1

        fail = 0
        possible_nodes= list(np.arange(G.N))
        picked_nodes = []

        for node in range(n):
            if len(possible_nodes) == 0:
                fail = 1
            else:
                pick = np.random.choice(possible_nodes)
                picked_nodes.append(pick)
                neighbors = np.where(A[pick,:])[0]
                possible_nodes = list( set(possible_nodes) - set(neighbors)  )

        if not fail:
            return picked_nodes
    
    print('Maximum tries reached. Reduce n')


def generate_anomaly(G, signal, anomaly_type=1):
    # signal is expected to be a matrix containing different signals in each column
    # each column will receive different anomalies

    if anomaly_type==1:
        anomaly_factor_min = 0.05
        anomaly_factor_max = 0.15
    elif anomaly_type==2:
        anomaly_factor_min = 0.10
        anomaly_factor_max = 0.20

    N = signal.shape[0]
    size = signal.shape[1]

    # Defining number of anomalous sensors per signal. Each column of signal has a different number of anomalies
    # This value is up to 5% the number of sensors (rounded up)
    percentage_per_signal = 0.05*np.random.rand(size) + 1e-5 # value added to avoid zero
    number_per_signal = np.ceil(percentage_per_signal*N)

    anomalous_positions = [pick_non_adjacent(G,int(n)) for n in number_per_signal]

    anomaly = np.zeros(signal.shape)
    for i in range(size):
        n = int(number_per_signal[i])
        anomalous_positions = pick_non_adjacent(G,n)

        anomaly[anomalous_positions, i] = signal[:,i].ptp()*np.random.uniform(low=anomaly_factor_min,
                                                                              high=anomaly_factor_max,
                                                                              size=n)*np.random.choice([+1,-1],size=n)
        
    label = np.zeros(signal.shape)
    label[np.where(anomaly)] = 1

    return anomaly, label

def generate_data(G, size, anomaly_type=1):    

    # Generating healthy data
    displacement = generate_smooth(G, size)
    ramp = generate_ramp(G, size)
    noise = np.sqrt(0.1)*np.random.randn(G.N,size) #noise power = 0.1, SNR = 20 dB

    signal = displacement + ramp + noise

    # Introducing anomalies
    anomaly, label = generate_anomaly(G, signal, anomaly_type)
    signal = signal + anomaly
    
    return signal, label

def hfilter(G, cut=2):
    L = G.L.toarray()
    w, V = np.linalg.eigh(L)
    wh = np.ones(G.N)
    wh[w<cut] = 0
    Hh = V @ np.diag(wh) @ V.T
    return Hh

def generate_cluster_anomaly(df, nodes, G, data_size=10, partition=20, anomaly_level=10, n_anomalies=1):

    nodes['cluster'] = KMeans(n_clusters=partition, n_init='auto').fit_predict(nodes[['northing','easting']])

    df.drop('cluster', axis=1, inplace=True)
    df = df.merge(nodes[['pid','cluster']], how='left', on='pid')

    X = []
    label = []
    df_anomaly_list = []

    for sample in range(data_size):
        df_anomaly = df[['timestamp','pid','cluster','smoothed']].copy()
        df_anomaly['anomaly'] = 0

        anomalous_clusters = np.random.choice(nodes.cluster.unique(), size=n_anomalies)

        for index, cluster in enumerate(anomalous_clusters):
            index = index+1

            anomaly_sensor = (df_anomaly.cluster==cluster)
            anomaly_period = (df_anomaly.timestamp>'Jul 2020')&(df_anomaly.timestamp<'Jan 2021')
            anomaly_loc = anomaly_sensor&anomaly_period

            df_anomaly.loc[anomaly_loc, 'smoothed'] += anomaly_level
            df_anomaly.loc[anomaly_loc, 'anomaly'] = index
        
        X.append(df_anomaly.pivot(index='pid', columns='timestamp', values='smoothed').values)
        label.append(df_anomaly.pivot(index='pid', columns='timestamp', values='anomaly').values.max(axis=1))
        df_anomaly_list.append(df_anomaly)

    X = np.array(X)
    label = np.array(label)

    return X, label, df_anomaly_list



def get_score(nodes, df_anomaly, S):

    nodes['pred'] = S.argmax(dim=1).cpu().numpy()
    nodes['score'] = S.softmax(dim=-1).detach().cpu().numpy().max(axis=1)
    nodes['anomaly'] = df_anomaly[['pid','anomaly']].groupby('pid').anomaly.max().values

    most_common_preds = nodes.query('anomaly!=0').groupby('anomaly')['pred'].apply(lambda x: x.mode()[0])

    nodes['new_pred'] = nodes['pred']
    nodes.loc[~nodes.pred.isin(most_common_preds.values),'new_pred'] = -1

    max_anomaly = nodes.groupby('new_pred')['anomaly'].transform('max')
    nodes.loc[nodes['new_pred'] != -1, 'new_pred'] = max_anomaly
    nodes.loc[nodes['new_pred'] == -1, 'new_pred'] = 0

    average = 'binary' if df_anomaly.anomaly.nunique()==2 else 'weighted'
    cluster_score = f1_score(y_true=nodes.anomaly, y_pred=nodes.new_pred, average=average)

    tpr, fpr, _ = roc_params(metric=nodes.score, label=(nodes.anomaly>0), interp=True)
    auc = compute_auc(tpr,fpr)

    return cluster_score, auc#, nodes



####################################
def main():
    return 0


if __name__ == "__main__":
    main()



