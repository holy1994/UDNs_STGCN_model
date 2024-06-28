import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utils
import torch
import torch.nn as nn
from tqdm import tqdm
from stgcn import STGCN
from joblib import Parallel, delayed
import gc

class Process():
    def __init__(self, A, X,epochs,train_ind,val_ind,test_ind, n,args):
        self.A = A
        self.X = X
        self.n = n
        self.train_ind =train_ind
        self.val_ind = val_ind
        self.test_ind = test_ind
        self.epochs = epochs
        self.args =args
        self.loss_criterion = nn.MSELoss()

    def train_epoch(self,net,optimizer,A_wave):
        """
        Trains one epoch with the given data.
        :param training_input: Training inputs of shape (num_samples, num_nodes,
        num_timesteps_train, num_features).
        :param training_target: Training targets of shape (num_samples, num_nodes,
        num_timesteps_predict).
        :param batch_size: Batch size to use during training.
        :return: Average loss for this epoch.
        """
        node_ind = list(map(int, self.train_ind))
        epoch_training_losses = []
        self.X['train_loader'].shuffle()
        for iter, (x, y) in enumerate(self.X['train_loader'].get_iterator()):
            net.train()
            optimizer.zero_grad()

            training_input = torch.Tensor(x.copy()).to(self.args.device)
            training_target = torch.Tensor(y.copy()).to(self.args.device)
            out = net(A_wave, training_input)

            label_array = utils.ind_array(node_ind, out.shape)

            pre = out * torch.Tensor(label_array).to(self.args.device)
            loss = self.loss_criterion(pre, training_target[:, :, :, 0])
            loss.backward()
            optimizer.step()

            epoch_training_losses.append(loss.detach().cpu().numpy())
        return sum(epoch_training_losses) / len(epoch_training_losses)

    def eval_epoch(self, net, A_wave, X, n, node_ind, dir_path, state='training_val'):
        outputs = []
        real = []
        for iter, (x, y) in enumerate(X.get_iterator()):
            with torch.no_grad():
                net.eval()
                testx = torch.Tensor(x.copy()).to(self.args.device)
                testy = torch.Tensor(y.copy()).to(self.args.device)

                out = net(A_wave, testx)
                outputs.append(out)
                real.append(testy)

        yhat = torch.cat(outputs, dim=0)
        realy = torch.cat(real, dim=0).squeeze()
        node_ind = list(map(int, node_ind))
        epoch_testing_loss = []
        epoch_testing_mae = []
        epoch_testing_mape = []
        for i in range(12):
            label_array = utils.ind_array(node_ind, yhat.shape)

            yhat_ = yhat[:, :, i]
            real_ = realy[:, :, i]

            pre_ = yhat_ * torch.Tensor(label_array[:, :, i]).to(self.args.device)
            rel_ = real_ * torch.Tensor(label_array[:, :, i]).to(self.args.device)
            test_loss = self.loss_criterion(pre_, rel_).to(device="cpu")
            epoch_testing_loss.append(test_loss.item())

            pre = utils.inverse_transform(yhat_, node_ind, os.path.join(self.args.dataset_dir, "scaler"))
            tru = utils.inverse_transform(real_, node_ind, os.path.join(self.args.dataset_dir, "scaler"))

            mape = np.mean(np.absolute(pre - tru) / np.absolute(tru)) * 100
            mae = np.mean(np.absolute(pre - tru))

            epoch_testing_mae.append(mae)
            epoch_testing_mape.append(mape)

            if i == 0 and state == 'testing':
                # pass
                utils.go_ploty(pre[:, 0],
                               tru[:, 0], dir_path,
                               f'svd_{n}_mape={mape :3.3f}%_node_{node_ind[0]}_unmeasured')
            elif i == 0 and state == 'training_val':
                pass
            else:
                continue
        # 在dim=0维度concatenate
        return (sum(epoch_testing_loss) / len(epoch_testing_loss),
                sum(epoch_testing_mae) / len(epoch_testing_mae), sum(epoch_testing_mape) / len(epoch_testing_mape))

    def process_iteration(self,process,idx):
        pbar = tqdm(total=10, desc='Generating walks (CPU: {})'.format(idx))

        templ = pd.DataFrame(columns=['save_percent','val_mape','seed','test_mape'])
        aim_mape = 1
        for processing in tqdm(process,desc='Iters',unit='processing'):
            pbar.update(1)
            if aim_mape>=0.06:
                dir_path0 = os.path.join(self.args.process_save, f'save_info_{self.n}')
                dir_path1 = os.path.join(dir_path0, f'processing{processing + 1}')
                dir_path2 = os.path.join(dir_path1, 'checkpoint')
                dir_path3 = os.path.join(dir_path1, 'picture')
                os.makedirs(dir_path1, exist_ok=True)
                os.makedirs(dir_path2, exist_ok=True)
                os.makedirs(dir_path3, exist_ok=True)
                print(f"Iter/{processing} 本次test_mape={aim_mape}  高于6%,继续训练")
                #加入随机种子
                # seed = np.random.randint(1000000)
                seed = 448671
                if self.n!=1:
                    temp_matrix_A = utils.svd_reduction(self.A,self.n,seed)
                    reduced_matrix_A = np.transpose(temp_matrix_A,(1,0))
                else:
                    reduced_matrix_A = self.A.copy()
                A_wave = torch.from_numpy(reduced_matrix_A)
                A_wave = A_wave.to(device=self.args.device).to(device=self.args.device)

                net = STGCN(A_wave.shape[1],A_wave.shape[0], self.X['x_train'].shape[3],
                                 12, 12).to(device=self.args.device)
                optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
                print("start training")
                training_losses = []
                validation_losses = []
                validation_maes = []
                validation_mapes = []
                for epoch in tqdm(range(self.epochs), desc="Training", unit="epoch"):
                    train_loss = self.train_epoch(net,optimizer,A_wave)
                    training_losses.append(train_loss)

                    val_loss,val_mae,val_mape = self.eval_epoch(net,A_wave,self.X['val_loader'],self.n,self.val_ind,dir_path1)
                    validation_losses.append(val_loss)
                    validation_maes.append(val_mae)
                    validation_mapes.append(val_mape)
                    print("Iter/{} Training loss: {}".format(processing,train_loss))
                    print("Iter/{} Validation loss: {}".format(processing,val_loss))
                    print("Iter/{} Validation MAE: {}".format(processing,val_mae))
                    print("Iter/{} Validation MAPE: {}%".format(processing,val_mape))
                    torch.save(net.state_dict(), dir_path2+"/_epoch_"+str(epoch)+"_"+str(round(val_loss, 2))+".pth")
                templ.at[processing,'val_mape'] = validation_mapes[-1]
                templ.at[processing, 'seed'] = seed
                templ.at[processing, 'save_percent'] = self.n

                print(f'保存{self.n}/{processing}信息 损失为{validation_mapes[-1] :3.3f}')
                bestid = np.argmin(validation_losses)
                print("start testing")
                # Run test
                net.load_state_dict(
                    torch.load(dir_path2+"/_epoch_" + str(bestid) + "_" + str(round(validation_losses[bestid], 2)) + ".pth"))
                test_loss,test_mae, test_mape = self.eval_epoch(net,A_wave,self.X['test_loader'],self.n,self.test_ind,dir_path1,'testing')
                torch.save(net.state_dict(),dir_path2+f"/dropout_epoch{self.epochs}"+"_exp"+"_best_"+str(round(validation_losses[bestid],2))+".pth")
                templ.at[processing, 'test_mape'] = test_mape
                aim_mape = test_mape/100
                print("Iter/{} Testing loss: {}".format(processing,test_loss))
                print("Iter/{} Testing MAE: {}".format(processing,test_mae))
                print("Iter/{} Testing MAPE: {}%".format(processing,test_mape))
                gc.collect()
            else:
                print(f"在第{processing}次，达到目标训练精度")
        pbar.close()
        return templ
