import os
import argparse
import random
import shutil
import numpy as np
import time
import utils
import torch
import torch.nn as nn
from tqdm import tqdm
from stgcn import STGCN
import pandas as pd
import gc
import sys
import itertools
import pickle

use_gpu = True
num_timesteps_input = 48
num_timesteps_output = 12
epochs = 20
process_size =10
batch_size = 64

parser = argparse.ArgumentParser(description='STGCN')
parser.add_argument('--enable-cuda', default=use_gpu,
                    help='Enable CUDA')
parser.add_argument('--dataset_dir', type=str, default='data/dataset', help='data path') # 数据集
parser.add_argument('--batch_size', type=int, default=64, help='batch size') # batch_size=64
# parser.add_argument('--components_test', type=int, default=[0.85], help='batch size') # batch_size=64
parser.add_argument("--unmeasured_node", type=int, default=1, help="unmeasured node data.", )
parser.add_argument('--process_save', type=str, default='results/process_folder', help='save process path')
parser.add_argument('--save', type=str, default='garage4', help='save path')
args = parser.parse_args()
args.device = None
if args.enable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')

os.makedirs(args.process_save,exist_ok=True)

def train_epoch(X,node_ind):
    """
    Trains one epoch with the given data.
    :param training_input: Training inputs of shape (num_samples, num_nodes,
    num_timesteps_train, num_features).
    :param training_target: Training targets of shape (num_samples, num_nodes,
    num_timesteps_predict).
    :param batch_size: Batch size to use during training.
    :return: Average loss for this epoch.
    """
    node_ind = list(map(int, node_ind))
    epoch_training_losses = []
    X['train_loader'].shuffle()
    for iter, (x, y) in enumerate(X['train_loader'].get_iterator()):
        net.train()
        optimizer.zero_grad()
        #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        x_ = utils.ind_array_copy(input_ind,x.shape)
        x_ = x_[:,:,np.newaxis]
        x_ = x_[:,:,:,np.newaxis]
        x_ = np.tile(x_, (1, 1,x.shape[2],x.shape[3]))
        x_x = x*x_
        #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        training_input = torch.Tensor(x_x).to(args.device)
        training_target = torch.Tensor(y).to(args.device)
        out = net(A_wave, training_input)

        label_array =utils.ind_array(node_ind,out.shape)

        pre = out*torch.Tensor(label_array).to(args.device)

        loss = loss_criterion(pre, training_target[:, :, :, 0])
        loss.backward()
        optimizer.step()

        epoch_training_losses.append(loss.detach().cpu().numpy())
    return sum(epoch_training_losses)/len(epoch_training_losses)
def sliding_inverse(row_data):
    data_head = row_data[:,:,0]
    data_tail = row_data[-1 :,:].squeeze().permute(1,0)
    data_to_ = torch.cat((data_head,data_tail),dim=0)
    return data_to_
def random_list(original_list,input_num):

    combinations = itertools.combinations(original_list, input_num)
    all_combinations = []

    for combo in combinations:
        a = list(combo)
        b = [x for x in original_list if x not in a]
        all_combinations.append((a, b))
    return all_combinations

def eval_epoch(X,processing,all_node,label_ind,dir_path,epoch,state='training_val'):
    outputs = []
    real = []
    for iter,(x,y) in enumerate(X.get_iterator()):
        with torch.no_grad():
            net.eval()

            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            x_ = utils.ind_array_copy(input_ind, x.shape)
            x_ = x_[:, :, np.newaxis]
            x_ = x_[:, :, :, np.newaxis]
            x_ = np.tile(x_, (1, 1, x.shape[2], x.shape[3]))
            x_x = x * x_
            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

            testx = torch.Tensor(x_x).to(args.device)
            testy = torch.Tensor(y).to(args.device)

            out = net(A_wave, testx)
            outputs.append(out)
            real.append(testy)

    yhat = torch.cat(outputs, dim=0)
    realy = torch.cat(real, dim=0).squeeze()

    label_ind = list(map(int, label_ind))
    all_node = list(map(int, all_node))

    yhat_ = sliding_inverse(yhat)
    real_ = sliding_inverse(realy)
    label_array = utils.ind_array_copy(label_ind,yhat_.shape)
    pre_ = yhat_*torch.Tensor(label_array).to(args.device)
    rel_ = real_*torch.Tensor(label_array).to(args.device)

    test_loss = loss_criterion(pre_, rel_).to(device="cpu")

    pre = utils.inverse_transform(yhat_,all_node,label_ind,os.path.join(args.dataset_dir, "scaler_all"))
    tru = utils.inverse_transform(real_,all_node,label_ind,os.path.join(args.dataset_dir, "scaler_all"))


    x_nse = 0
    node_mape_ = pd.DataFrame(columns=label_ind)
    node_mae_ = pd.DataFrame(columns=label_ind)
    for num_, node_ in enumerate(label_ind):
        mape_ = np.mean(np.absolute(pre[:, num_] - tru[:, num_]) / np.absolute(tru[:, num_])) * 100
        mae_ = np.mean(np.absolute(pre[:, num_] - tru[:, num_]))
        node_mape_.at[processing, node_] = mape_
        node_mae_.at[processing, node_] = mae_

    node_mape_sorted = node_mape_.sort_index(axis=1)
    node_mae_sorted = node_mae_.sort_index(axis=1)
    #
    inf_positions = node_mape_.isin([np.inf, -np.inf])
    columns_with_inf = inf_positions.any().index[inf_positions.any()].tolist()
    del_inf_mape = node_mape_sorted.drop(columns=columns_with_inf, inplace=False)
    del_inf_mae = node_mae_sorted.drop(columns=columns_with_inf, inplace=False)
    mape = del_inf_mape.iloc[0, :].mean()
    mae = del_inf_mae.iloc[0, :].mean()
    if state == 'testing':
        accuracy_ = [(np.count_nonzero(del_inf_mape.values <= str) / del_inf_mape.shape[1]) for str in [5, 10, 20]]
        accuracy_.append(np.count_nonzero(del_inf_mape.values > 20) / del_inf_mape.shape[1])
        accuracyz_table.loc[processing, :] = accuracy_
        pre_out = pd.DataFrame(pre,columns=label_ind)
        out_pre = pre_out.sort_index(axis=1)
        tru_out = pd.DataFrame(tru,columns=label_ind)
        out_tru = tru_out.sort_index(axis=1)
        mapped_test_label = sorted(label_ind).index(349)
        utils.go_ploty(out_pre.iloc[:,mapped_test_label],
                       out_tru.iloc[:,mapped_test_label],dir_path,
                       f'svd_mape={node_mape_sorted.iloc[0,mapped_test_label] :3.3f}%_node_349_unmeasured')

        test_dict = {'predict':out_pre,'truth':out_tru}
        with open(os.path.join(dir_path, 'dataset_test.pkl'), 'wb') as f:
            pickle.dump(test_dict, f, protocol=2)

    elif state == 'training_val':
        pass

    return (test_loss.item(),mae,mape,x_nse)

def copy_custom_module_files(modules, destination_folder):
    for module_name in modules:
        module = sys.modules.get(module_name)
        if module and hasattr(module, '__file__'):
            module_file = module.__file__
            shutil.copy(module_file, destination_folder)
def set_seed(seed):
    torch.manual_seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
def out_txt(txt,output_dir,write_name):
    out_txt = txt.copy()
    out_txt = list(map(str, out_txt))
    out_txt = ','.join(out_txt)
    with open(os.path.join(output_dir, f'{write_name}.txt'), 'w') as file:
        file.write(out_txt)
def moedl_start():

    torch_seed = 660172
    set_seed(torch_seed)
    current_script = os.path.realpath(__file__)
    script_path = __file__
    modules = ['stgcn', 'utils']
    destination_folder = args.process_save
    shutil.copy(current_script, destination_folder)
    copy_custom_module_files(modules, destination_folder)
    # 模型实例化>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    n = 0.85
    A = utils.load_A(args.dataset_dir)
    A_outs = utils.get_normalized_adj(A)
    seed = 724296
    temp_matrix_A = utils.svd_reduction(A_outs, n, seed)
    reduced_matrix_A = np.transpose(temp_matrix_A, (1, 0))
    A_wave = torch.from_numpy(reduced_matrix_A)
    A_wave = A_wave.to(device=args.device).to(device=args.device)
    net = STGCN(len(A), A_wave.shape[0],
                2, num_timesteps_input,
                num_timesteps_output).to(device=args.device)

    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
    loss_criterion = nn.MSELoss()
    return n,net,optimizer,loss_criterion,torch_seed,seed,A_wave
def random_number(int,out):
    random_combinations = utils.loda_ind_label(dataset_path)[:-1]
    random.shuffle(random_combinations)
    selected_three = random_combinations[:int]
    remaining = [item for item in random_combinations if item not in selected_three]
    random_remaining = remaining[:out]
    return selected_three,random_remaining
def test_process():
    print("start testing")
    net.load_state_dict(
        torch.load(dir_path2 + "/_epoch_" + f'{epochs - 1}' + "_" + str(
            round(validation_losses[epochs - 1], 2)) + ".pth"))  ###

    test_loss, test_mae, test_mape, nse = eval_epoch(X['test_loader'], processing, all_node, test_ind, dir_path1, None,
                                                     'testing')
    torch.save(net.state_dict(), dir_path2 + f"/dropout_epoch{epochs}" + "_exp" + f"_{epochs - 1}" + ".pth")

    templ.loc[processing, dataframe_columns] = [data_num, validation_mapes[-1],
                                                test_mape, seed, torch_seed,
                                                len(input_ind),
                                                len(out_ind)]
    templ.to_excel(os.path.join(dir_path1, 'templ.xlsx'), index=False)
    print("Iter/{} Testing loss: {}".format(processing, test_loss))
    print("Iter/{} Testing MAE: {}".format(processing, test_mae))
    print("Iter/{} Testing MAPE: {}%".format(processing, test_mape))

if __name__ == '__main__':
    # # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    dataframe_columns = ['dataset_number','val_mape','test_mape','seed','torch_seed','输入点个数','输出点个数']
    results = pd.DataFrame(columns=dataframe_columns)
    val_mean = []
    for data_num in range(1,7):
        dataset_path = os.path.join(args.dataset_dir,f'dataset{data_num}')
        random_combinations = utils.load_pickle(os.path.join(dataset_path, 'random_combinations.pkl'))
        for iter_num in range(1,11):
            mian_path = os.path.join(args.process_save,f'iter_{iter_num}')
            n, net, optimizer, loss_criterion, torch_seed, seed, A_wave = moedl_start()
            start_time = time.time()
            dir_path0 = os.path.join(mian_path, f'0.85%train_dataset{data_num}')###删除后缀
            templ = pd.DataFrame(columns=dataframe_columns)
            accuracyz_table = pd.DataFrame(columns=['acc<=5%', 'acc<=10%', 'acc<=20%', "acc>20%"])
            for processing in tqdm(range(process_size),desc='Iters',unit='processing'):
                dir_path1 = os.path.join(dir_path0, f'processing{processing+1}')
                dir_path2 = os.path.join(dir_path1,'checkpoint')
                dir_path3 = os.path.join(dir_path1,'picture')
                os.makedirs(dir_path1, exist_ok=True)
                os.makedirs(dir_path2, exist_ok=True)
                os.makedirs(dir_path3, exist_ok=True)

                if processing!=0:
                    net.load_state_dict(
                        torch.load(os.
                                   path.join(dir_path0, f'processing{processing}','checkpoint',
                                            f"_epoch_{epochs-1}_"+str(round(validation_losses[epochs-1], 2))+".pth")))
                print("start training")

                training_losses = []
                validation_losses = []
                validation_maes = []
                validation_mapes = []
                testing_losses = []
                testing_maes = []
                ix = np.random.randint(7)
                print(ix)
                input_ind = random_combinations[ix][0]
                out_ind = random_combinations[ix][1]
                print(input_ind)
                print(out_ind)
                out_txt(input_ind, dir_path1, 'write_input')
                out_txt(out_ind, dir_path1, 'write_out')
                for epoch in tqdm(range(epochs), desc="Training", unit="epoch"):
                    X, node_ind,all_node = utils.load_metr_la_data(ix, dataset_path, args.batch_size,
                                                                   args.batch_size, args.batch_size)
                    test_ind = list(set(all_node)-set(node_ind[:-1]))
                    train_loss = train_epoch(X,out_ind)

                    training_losses.append(train_loss)
                    val_loss,val_mae,val_mape,nse = eval_epoch(X['val_loader'],n,all_node,out_ind,dir_path1,epoch)
                    validation_losses.append(val_loss)
                    validation_maes.append(val_mae)
                    validation_mapes.append(val_mape)
                    print("Iter/{} Training loss: {}".format(processing,train_loss))
                    print("Iter/{} Validation loss: {}".format(processing,val_loss))
                    print("Iter/{} Validation MAE: {}".format(processing,val_mae))
                    print("Iter/{} Validation MAPE: {}%".format(processing,val_mape))

                    torch.save(net.state_dict(), dir_path2+"/_epoch_"+str(epoch)+"_"+str(round(val_loss, 2))+".pth")
                    gc.collect()

                bestid = np.argmin(validation_losses)
                with open(os.path.join(dir_path1, f'best_loss.txt'), 'w') as file:
                    file.write(str(bestid))
                np.save(os.path.join(dir_path1, 'training_losses.npy'), training_losses)
                np.save(os.path.join(dir_path1,'validation_losses.npy'),validation_losses)
                test_process()
                del X
                gc.collect()
            accuracyz_table.to_excel(os.path.join(dir_path0,'各节点计算精度表.xlsx'), index=False)
            val_mean.append(templ['val_mape'].mean())
            results = pd.concat([results,templ], ignore_index=True)
            end_time = time.time()
            print(f'TimeUsed: {end_time - start_time:.3f}/10 Processing    \n')

        results.to_excel(os.path.join(mian_path, '不同输入数据对比表.xlsx'), index=False)




