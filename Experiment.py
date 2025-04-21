import torch
from Datasets import MNISTDataset
from Encoding import RateEncoder, TtfsEncoder, DirectEncoder, PoissonGen
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
        elif self.config['encoder'] == "poisson":
            return PoissonGen(self.config['num_steps'])
        elif self.config['encoder'] == "ttfs":
            return TtfsEncoder(self.config['num_steps'])
        elif self.config['encoder'] == "direct":
            return DirectEncoder(self.config['num_steps'])

    def init_decoder(self):
        decoder_type = self.config['decoder']
        params = self.config['decoder_params'].get(decoder_type, {})

        if decoder_type == "rate":
            from Decoding import RateDecoder
            return RateDecoder(self.config['num_steps'], **params)
        elif decoder_type == "latency":
            from Decoding import LatencyDecoder
            return LatencyDecoder(self.config['num_steps'], **params)
        elif decoder_type == "first_spike":
            from Decoding import FirstSpikeDecoder
            return FirstSpikeDecoder(self.config['num_steps'], **params)
        else:
            raise ValueError(f"Decoder no soportado: {decoder_type}")

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
