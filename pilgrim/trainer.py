import torch
import os
import time
import pandas as pd
import schedulefree
import math

from .utils import generate_random_walks

class Trainer:
    def __init__(self, net, num_epochs, device, batch_size=10000, lr=0.001, name="", K_min=1, K_max=55, all_moves=None, inverse_moves=None, V0=None):
        self.net = net.to(device)
        self.lr = lr
        self.device = device
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.criterion = torch.nn.MSELoss()
        self.optimizer = schedulefree.AdamWScheduleFree(self.net.parameters(), lr=lr)
        self.epoch = 0
        self.id = int(time.time())
        self.log_dir = "logs"
        self.weights_dir = "weights"
        self.name = name
        self.K_min = K_min
        self.K_max = K_max
        self.walkers_num = 1_000_000 // self.K_max
        self.all_moves = all_moves
        self.inverse_moves = inverse_moves
        self.V0 = V0
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.weights_dir, exist_ok=True)

    def _train_epoch(self, X, Y):
        self.net.train()
        avg_loss = 0.0
        total_batches = X.size(0) // self.batch_size
        
        for i in range(0, X.size(0), self.batch_size):
            data = X[i:i + self.batch_size]
            target = Y[i:i + self.batch_size]
            output = self.net(data)
            loss = self.criterion(output, target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            avg_loss += loss.item()

        return avg_loss / total_batches if total_batches > 0 else avg_loss

    def run(self):
        for epoch in range(self.num_epochs):
            self.epoch += 1

            # Data generation
            data_gen_start = time.time()
            X, Y = generate_random_walks(self.V0, self.V0.size(1), self.all_moves, self.inverse_moves, k=self.walkers_num, K_min=self.K_min, K_max=self.K_max)
            data_gen_time = time.time() - data_gen_start

            # Training step
            epoch_start = time.time()
            train_loss = self._train_epoch(X, Y.float())
            epoch_time = time.time() - epoch_start

            log_file = f"{self.log_dir}/{self.name}_{self.id}.csv"
            log_data = pd.DataFrame([{
                'epoch': self.epoch, 
                'train_loss': train_loss, 
                'vertices_seen': X.size(0),
                'data_gen_time': data_gen_time,
                'train_epoch_time': epoch_time
            }])
            log_data.to_csv(log_file, mode='a', header=not os.path.exists(log_file), index=False)

            # Save weights on powers of two
            if (self.epoch & (self.epoch - 1)) == 0:
                log2_epoch = int(math.log2(self.epoch))
                weights_file = f"{self.weights_dir}/{self.name}_{self.id}_e2pow{log2_epoch}.pth"
                torch.save(self.net.state_dict(), weights_file)

            # Save weights at 10,000 and 50,000 epochs
            if self.epoch in [10000, 50000]:
                weights_file = f"{self.weights_dir}/{self.name}_{self.id}_e{self.epoch}.pth"
                torch.save(self.net.state_dict(), weights_file)

        # Save final weights
        final_weights_file = f"{self.weights_dir}/{self.name}_{self.id}_e{self.epoch}_final.pth"
        torch.save(self.net.state_dict(), final_weights_file)