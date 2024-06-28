import os
import zipfile
import numpy as np
import torch
import pickle
import torch.nn as nn
import dask
import dask.array as da
import plotly.graph_objects as go
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
import networkx as nx
import joblib
from scipy.linalg import fractional_matrix_power
from sklearn.decomposition import PCA
import umap
def go_ploty(pre_,rea_,dir_path,pic_name):

    x = list(range(1,len(pre_)+1))
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=x, y=pre_, mode='lines', name='pred'))
    fig.add_trace(go.Scatter(x=x, y=rea_, mode='lines', name='real'))

    fig.update_layout(title=f'{pic_name} Plot',
                      xaxis_title='X-axis',
                      yaxis_title='Y-axis',
                      showlegend=True)
    # fig.show()
    fig.write_html(dir_path+f"/picture/{pic_name}_plot.html")

def svd_reduction(adj_matrix,variance_threshold,seed):
    if variance_threshold != 1:
        svd = TruncatedSVD(n_components=n_componet(adj_matrix,variance_threshold),random_state=seed)
        if len(adj_matrix.shape) > 2:
            reduce_matrixs = []
            for i in range(len(adj_matrix)):
                reduced_matrix_ = svd.fit_transform(np.transpose(adj_matrix[i,:,:,0],(1,0)))
                reduce_matrixs.append(np.transpose(reduced_matrix_,(1,0)))
        else:
            reduced_matrix_A = svd.fit_transform(np.array(adj_matrix))
    else:
        reduced_matrix_A = adj_matrix.copy()

    return reduced_matrix_A

def n_componet(adj_matrix,variance_threshold):
    pca = PCA().fit(np.array(adj_matrix))

    explained_variance_ratio = pca.explained_variance_ratio_

    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

    components_for_threshold = np.where(cumulative_variance_ratio >= variance_threshold)[0][0] + 1
    return components_for_threshold

def umap_reduction(adj_matrix,variance_threshold,seed):

    reducer = umap.UMAP(n_components=n_componet(adj_matrix,variance_threshold),random_state=seed)

    embedding = reducer.fit_transform(np.array(adj_matrix))
    return embedding
def pca_reduction(adj_matrix,variance_threshold,seed):
    pca = PCA(n_components=variance_threshold,random_state=seed)

    pca.fit(np.array(adj_matrix))

    X_reduced = pca.transform(np.array(adj_matrix))
    return X_reduced
def load_A(path):
    nodes_idex, adj_matrix_node, adj_matrix = load_pickle(os.path.join(path,'adj_mat.pkl'))
    A = adj_matrix
    return A
def load_metr_la_data(num,dataset_dir, batch_size, valid_batch_size= None, test_batch_size= None):
    data= {}
    for category in ['train', 'val', 'test']:
        cat_data_x = load_zarr(os.path.join(dataset_dir, f'x_{category}_{num}.zarr'))
        cat_data_y = load_zarr(os.path.join(dataset_dir, f'y_{category}_{num}.zarr'))
        data['x_' + category] = cat_data_x.astype(np.float16)
        data['y_' + category] = cat_data_y.astype(np.float16)
    data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size)
    data['val_loader'] = DataLoader(data['x_val'], data['y_val'], valid_batch_size)
    data['test_loader'] = DataLoader(data['x_test'], data['y_test'], test_batch_size)

    with open(os.path.join(dataset_dir,'train_label_index.txt')) as f:
        load_txt = f.read().strip().split(',')
    with open(os.path.join(dataset_dir, 'all_index_index.txt')) as f:
        all_txt = f.read().strip().split(',')

    return data,load_txt,all_txt
def loda_ind_label(label_dir):
    with open(os.path.join(label_dir,'train_label_index.txt')) as f:
        label_ind = f.read().strip().split(',')
    label_ind = list(map(int, label_ind))
    return label_ind
def assign_color(value):
    if value <= -2:
        return 'orange'
    if value <= -1:
        return 'red'
    if value <= 5:
        return '#BF81d0'
    elif value <= 10:
        return '#547ac0'
    elif value <= 20:
        return '#79cb9b'
    else:
        return '#898988'
def nx_plot(node_mape_,label_index,out_ind,args):

    G = nx.read_gexf(os.path.join(args.dataset_dir, "graph.gexf"))

    nodes_mape_attributes_ = node_mape_.iloc[0, :].to_dict()
    nodes_mape_attributes = {str(key): value for key, value in nodes_mape_attributes_.items()}


    label_dict = {f'{i}': -1 for i in label_index}
    out_dict = {f'{n}': -2 for n in out_ind}
    nodes_mape_attributes.update(label_dict)
    nodes_mape_attributes.update(out_dict)

    nx.set_node_attributes(G, nodes_mape_attributes, 'attributes_mape')

    attributes_values = nx.get_node_attributes(G, 'attributes_mape')

    colors = [attributes_values.get(node, 100) for node in G.nodes()]
    color_map = [assign_color(value_) for value_ in colors]
    pic_path = os.path.join(args.process_save, 'network_picture')
    os.makedirs(pic_path,exist_ok=True)
    for node in G.nodes():
        pos_str = G.nodes[node]['pos']
        x_str, y_str = pos_str.split(',')
        G.nodes[node]['pos'] = (float(x_str), float(y_str))


    pos = nx.get_node_attributes(G, 'pos')

    nx.draw(G, pos, node_size=20, node_color=color_map, arrows=False)
    plt.savefig(os.path.join(pic_path, f'test_input{label_index[0]}_out{out_ind[0]}.png'), dpi=300)
    plt.show()
def inverse_transform(data,all_node,node_ind,scaler_path):
    scaler_ = joblib.load(scaler_path)
    inverse_data = data[:, node_ind]
    inverse_data = inverse_data.detach().cpu().numpy().reshape(-1, len(node_ind))
    data_ = np.ones((len(inverse_data),531), dtype=np.float16)

    all_node = sorted(all_node)
    mapped_node_ind = []
    for node_id in node_ind:
        mapped_node_ind.append(all_node.index(node_id))
    data_[:,mapped_node_ind]=inverse_data

    inverse_data_ = scaler_.inverse_transform(data_)[:,mapped_node_ind]

    return inverse_data_
def ind_array(node_ind,shape):
    label = np.zeros((shape[1], 1))
    label[[list(map(int, node_ind))], :] = 1
    label = np.tile(label, (1, shape[-1]))
    label = label[np.newaxis, :, :]
    label_array = np.tile(label, (shape[0], 1, 1))
    return label_array
def ind_array_copy(node_ind,shape):
    label = np.zeros((shape[1], 1))
    label[[list(map(int, node_ind))], :] = 1
    label = np.tile(label.T, (shape[0],1))
    return label
class DataLoader(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True):

        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                yield (x_i, y_i)
                self.current_ind += 1

        return _wrapper()
class NSELoss(nn.Module):
    def __init__(self):
        super(NSELoss, self).__init__()

    def forward(self, y_pred, y_true,ind):

        y_pred = y_pred[:,ind,...]
        y_true = y_true[:,ind,...]

        y_mean = torch.mean(y_true, dim=0)

        numerator = torch.sum((y_true - y_pred)**2, dim=0)

        denominator = torch.sum((y_true - y_mean)**2, dim=0)

        epsilon = 1e-80
        denominator = torch.where(denominator < epsilon, epsilon, denominator)

        nse_loss = 1 - numerator / denominator

        return 1-torch.mean(nse_loss)

class StandardScaler():

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self,data):
        return (data-self.mean) / self.std

    def inverse_transform(self,data):
        return (data*self.std) + self.mean

def calculate_row_proportions(row):
    total_count = len(row)
    proportions = {
        'acc<=5%': sum(row <= 5) / total_count,
        'acc<=10%': sum(row <= 10) / total_count ,
        'acc<=20%': sum(row <= 20) / total_count,
        'acc>20%': sum(row > 20) / total_count
    }
    return proportions
class MinMaxSacler():

    def __init__(self,maxx,minn):
        self.maxx = maxx
        self.minn = minn

    def transform(self,data):
        data_std = (data - data.min) / (data.max - data.min)
        self.data_max = data.max
        self.data_min = data.min
        return data_std * (self.maxx - self.minn) + self.minn

    def inverse_transform(self,data):
        data_inv = (data-self.minn)/(self.maxx-self.minn)
        return data_inv*(self.data_max-self.data_min)+self.data_min

def read_file(file_path):
    return da.from_zarr(file_path)
def load_zarr(file_path):
    read_task = read_file(file_path)
    dask_array = read_task.compute()
    return dask_array

def load_dataset(dataset_dir, batch_size, valid_batch_size= None, test_batch_size= None):
    data= {}
    for category in ['train', 'val', 'test']:
        cat_data_x = load_zarr(os.path.join(dataset_dir, 'x_'+category + '.zarr'))
        cat_data_y = load_zarr(os.path.join(dataset_dir, 'y_'+category + '.zarr'))
        data['x_' + category] = cat_data_x
        data['y_' + category] = cat_data_y

        scaler = StandardScaler(mean=data['x_' + category][..., 0].mean(), std=data['x_' + category][..., 0].std())
        data['x_' + category][..., 0] = scaler.transform(data["x_" + category][..., 0])

    data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size)
    data['val_loader'] = DataLoader(data['x_val'], data['y_val'], valid_batch_size)
    data['test_loader'] = DataLoader(data['x_test'], data['y_test'], test_batch_size)
    data['scaler'] = scaler
    return data
def load_pickle(pickle_file):
    try:
        with open(pickle_file,'rb') as f:
            pickle_data=pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data', pickle_file, ':', e)
        raise
    return pickle_data
def get_normalized_adj(A):

    A = A.astype(np.float32)

    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D = np.diag(D)
    D_half_neg = fractional_matrix_power(D,-0.5)
    A_wave = D_half_neg @ A @ D_half_neg

    return A_wave


def generate_dataset(X, num_timesteps_input, num_timesteps_output):
    """
    Takes node features for the graph and divides them into multiple samples
    along the time-axis by sliding a window of size (num_timesteps_input+
    num_timesteps_output) across it in steps of 1.
    :param X: Node features of shape (num_vertices, num_features,
    num_timesteps)
    :return:
        - Node features divided into multiple samples. Shape is
          (num_samples, num_vertices, num_features, num_timesteps_input).
        - Node targets for the samples. Shape is
          (num_samples, num_vertices, num_features, num_timesteps_output).
    """

    indices = [(i, i + (num_timesteps_input + num_timesteps_output)) for i
               in range(X.shape[2] - (
                num_timesteps_input + num_timesteps_output) + 1)]

    features, target = [], []
    for i, j in indices:
        features.append(
            X[:, :, i: i + num_timesteps_input].transpose(
                (0, 2, 1)))
        target.append(X[:, 0, i + num_timesteps_input: j])

    return torch.from_numpy(np.array(features)), \
           torch.from_numpy(np.array(target))
