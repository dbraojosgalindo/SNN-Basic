import torch
from Encoding import Encoder
from Decoding import Decoder
import torch.nn as nn

class Trainer:
    def __init__(self, net:nn.Module, optimizer, loss_fn, device, encoder:Encoder, decoder:Decoder, num_steps:int):
        self.net = net
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.encoder = encoder
        self.decoder = decoder
        self.num_steps = num_steps

    def train_epoch(self, train_loader):
        self.net.train()

        for data, targets in train_loader:
            data = self.encoder.encode(data).to(self.device)
            targets = targets.to(self.device)

            spk_rec, mem_rec = self.net(data.view(self.num_steps, data.size(1), -1), self.num_steps)

            loss_val = torch.zeros((1), dtype=torch.float, device=self.device)
            for step in range(self.num_steps):
                loss_val += self.loss_fn(mem_rec[step], targets)

            self.optimizer.zero_grad()
            loss_val.backward()
            self.optimizer.step()

    def evaluate(self, test_loader):
        self.net.eval()
        total = 0
        correct = 0

        with torch.no_grad():
            for data, targets in test_loader:
                data = self.encoder.encode(data).to(self.device)
                targets = targets.to(self.device)

                spk_rec, _ = self.net(data.view(self.num_steps, data.size(1), -1), self.num_steps)
                decoded = self.decoder.decode(spk_rec)

                _, predicted = decoded.max(1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        accuracy = 100 * correct / total
        #print(f"Test Set Accuracy: {accuracy:.2f}%")
        return accuracy
