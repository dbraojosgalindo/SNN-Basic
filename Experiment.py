import torch
from Datasets import MNISTDataset
from Encoding import RateEncoder, TtfsEncoder
from Decoding import RateDecoder
from Architecture import TwoLayerSNN
from Trainer import Trainer
import torch.nn as nn


class SNNExperiment:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.dataset = self.init_dataset()
        self.encoder = self.init_encoder()
        self.decoder = self.init_decoder()
        self.architecture = self.init_architecture()
        self.net = self.architecture.build_network().to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=config['lr'], betas=config['betas'])
        self.trainer = Trainer(self.net, self.optimizer, nn.CrossEntropyLoss(), self.device, self.encoder, self.decoder, self.config['num_steps'])

    def init_dataset(self):
        if self.config['dataset'] == 'MNIST':
            return MNISTDataset(self.config['data_path'], self.config['batch_size'])

    def init_encoder(self):
        if self.config['encoder'] == "rate":
            return RateEncoder(self.config['num_steps'], self.config.get('gain', 0.5))
        if self.config['encoder'] == "ttfs":
            return TtfsEncoder(self.config['num_steps'])

    def init_decoder(self):
        return RateDecoder()

    def init_architecture(self):
        return TwoLayerSNN(
            self.config['num_inputs'],
            self.config['num_hidden'],
            self.config['num_outputs'],
            self.config['beta']
        )

    def run(self):
        train_loader, test_loader = self.dataset.get_loaders()

        for epoch in range(self.config['num_epochs']):
            print(f"Epoch {epoch + 1}/{self.config['num_epochs']}")
            self.trainer.train_epoch(train_loader)

            if (epoch + 1) % self.config['eval_freq'] == 0:
                accuracy = self.trainer.evaluate(test_loader)

        print(f"\ndataset: {self.config['dataset']}. Encoder: {self.config['encoder']}. Architecture: {self.config['architecture']}. Decoder: {self.config['decoder']}")
        return accuracy
