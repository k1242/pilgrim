import torch
import torch.nn as nn

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from ipywidgets import IntProgress
from IPython.display import display, clear_output
import time
import wandb

class Meter:
    def __init__(self):
        self.metrics = {}
    
    def update(self, loss: float) -> None:
        """
        Update the meter with the new loss value.

        :param loss: Loss value to be added to the metrics.
        """
        self.metrics['loss'] += loss
    
    def init_metrics(self) -> None:
        """
        Initialize/reset the metrics.
        """
        self.metrics['loss'] = 0
        
    def get_metrics(self) -> dict:
        """
        Get the current metrics.

        :return: Dictionary of metrics.
        """
        return self.metrics

def get_dataloader(XY: tuple, batch_size: int) -> DataLoader:
    """
    Create a DataLoader from input data and targets.

    :param XY: Tuple of input data and targets.
    :param batch_size: Batch size for the DataLoader.
    :return: DataLoader object.
    """
    dataset = TensorDataset(*XY)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

class Trainer:
    def __init__(self, net: nn.Module, num_epochs: int, device: torch.device, XY: tuple, wandb_project_name: str = "Pilgrim_v1", verbose: int = 2, batch_size: int = 1024, wandb_flag: bool = False, add_name: str = "", version: int = 0, lr: float = 0.001, weights2load: str = False):
        """
        Trainer class for training the model.

        :param net: Neural network model to be trained.
        :param num_epochs: Number of training epochs.
        :param device: Device to perform computations (e.g., 'cuda', 'cpu').
        :param XY: Tuple containing training and validation data and targets.
        :param wandb_project_name: Name of the WandB project.
        :param verbose: Verbosity level.
        :param batch_size: Batch size for training.
        :param wandb_flag: Flag to indicate whether to use WandB.
        :param add_name: Additional name for the run.
        :param version: Version of the model.
        :param lr: Learning rate.
        :param weights2load: Path to pre-trained weights.
        """
        net_local = net 
        if weights2load: 
            net_local.load_state_dict(torch.load(weights2load, map_location=device))
            print("weights loaded")
        self.net = net_local.to(device)
        self.device = device
        self.wandb_project_name = wandb_project_name
        self.XY = XY
        self.train_size = XY[0].shape[0]
        self.val_size = XY[2].shape[0]   
        self.num_epochs = num_epochs
        self.verbose = verbose
        self.batch_size = batch_size
        self.wandb_flag = wandb_flag
        self.add_name = add_name
        self.version = version
        self.epoch = 0
        self.outputs = []
        self.targets = []
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.net.parameters()), lr=lr)
        self.best_loss = float('inf')
        self.phases = ['train', 'val']
        self.dataloaders = {
            'train': get_dataloader(XY[:2], batch_size),
            'val': get_dataloader(XY[2:], batch_size)
        }
        self.train_df_logs = pd.DataFrame()
        self.val_df_logs = pd.DataFrame()
    
    def init_dataloaders(self) -> None:
        """
        Initialize the dataloaders for training and validation.
        """
        self.dataloaders = {
            'train': get_dataloader(self.XY[:2], self.batch_size),
            'val': get_dataloader(self.XY[2:], self.batch_size)
        }

    def print_logs(self) -> None:
        """
        Print the training logs.
        """
        clear_output(wait=True)
        
        y_pred = np.concatenate(self.outputs, axis=0)
        y_true = np.concatenate(self.targets, axis=0)  

        if self.wandb_flag:
            wandb_log = {
                "epoch": self.epoch,
                "t_loss": self.train_df_logs.iloc[-1]['loss'],
                "v_loss": self.val_df_logs.iloc[-1]['loss'],
                "best_loss": self.best_loss
            }
            wandb.log(wandb_log)
        if self.verbose > 0:
            print(f"v{self.version}")
            print(f"epoch: {self.train_df_logs.shape[0]}", end="\n\n")
            print(f"train:")
            print(f"    loss    {self.train_df_logs.iloc[-1]['loss']:.3f}")

            print(f"val:")
            print(f"    loss    {self.val_df_logs.iloc[-1]['loss']:.3f}")
    
    def _train_epoch(self, phase: str) -> float:    
        """
        Train or validate the model for one epoch.

        :param phase: Training phase ('train' or 'val').
        :return: Loss value for the epoch.
        """
        if phase == 'train': 
            self.net.train()
        else:  
            self.net.eval()

        meter = Meter()
        meter.init_metrics()
        
        if self.verbose > 1:
            print(f"\n{phase}:")
            max_progress = self.train_size//self.batch_size if phase == 'train' else self.val_size//self.batch_size
            f = IntProgress(min=0, max=max_progress)
            display(f)
                
        self.outputs = []
        self.targets = []
        for i, (data, target) in enumerate(self.dataloaders[phase]):
            data = data.to(self.device)
            target = target.to(self.device)
            output = self.net(data).flatten()    
            loss = self.criterion(output, target)
                        
            if phase == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            else:
                self.outputs.append(output.detach().cpu().numpy())
                self.targets.append(target.detach().cpu().numpy())
                
            meter.update(loss.item())
            if self.verbose > 1: f.value += 1
                  
        metrics = meter.get_metrics()
        metrics = {k: v / (i + 1) for k, v in metrics.items()}
        df_logs = pd.DataFrame([metrics])
        
        if phase == 'train':
            self.train_df_logs = pd.concat([self.train_df_logs, df_logs], axis=0)
        else:
            self.val_df_logs = pd.concat([self.val_df_logs, df_logs], axis=0)   
        return loss
    
    def run(self) -> None:
        """
        Run the training process.
        """
        print("start")
        
        if self.wandb_flag:
            wandb.init(
                project=self.wandb_project_name, 
                name=f"{self.add_name}",
                tags=[self.add_name],
                config={
                    "v": self.version,
                    "batch_size": self.batch_size,
                    "lr": self.lr,
                    "hd1": self.net.hd1,
                    "hd2": self.net.hd2,
                    "nrd": self.net.nrd
                })
            print("wandb ok")
        
        for epoch in range(self.num_epochs):
            self.epoch += 1
            t1 = time.time()
            self._train_epoch(phase='train')
            with torch.no_grad():
                val_loss = self._train_epoch(phase='val')
            
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                torch.save(self.net.state_dict(), f"weights/{self.add_name}_{self.version}.pth")
            t2 = time.time()
        
            self.print_logs()
            if self.verbose > 0:
                print(f"\nremaining time: {(t2 - t1) * (self.num_epochs - epoch) / 60:.2f} min")
                print(f"time/epoch: {(t2 - t1) / 60:.2f} min")
        if self.wandb_flag: wandb.finish()
        print("end")
