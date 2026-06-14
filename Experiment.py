# Experiment.py
# Define la clase SNNExperiment, que orquesta la ejecución de un experimento SNN completo.

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import torch
from Datasets import MNISTDataset
from Encoding import *
from Decoding import RateDecoder, FirstSpikeDecoder, LatencyDecoder, PopulationRateDecoder, RankOrderDecoder, AllDecoders
from Architecture import TwoLayerSNN
from Trainer import Trainer
import torch.nn as nn


class SNNExperiment:
   def __init__(self, config):
       """
       Inicializa la SNN con la configuración dada.
       Prepara dataset, encoder, decoder, arquitectura, red, optimizador y trainer.
       """
       self.config = config
       # Selecciona el dispositivo (GPU si está disponible, si no CPU)
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
       encoder_type = self.config['encoder']
       params = self.config['encoder_params'].get(encoder_type, {})
       if self.config['encoder'] == "rate":
           return RateEncoder(self.config['num_steps'], **params)
       elif self.config['encoder'] == "poisson":
           return PoissonGen(self.config['num_steps'], **params)
       elif self.config['encoder'] == "ttfs":
           return TtfsEncoder(self.config['num_steps'], **params)
       elif self.config['encoder'] == "direct":
           return DirectEncoder(self.config['num_steps'])
       elif self.config['encoder'] == "delta":
           return DeltaEncoder(self.config['num_steps'], **params)
       elif self.config['encoder'] == "MW":
           return MWEncoder(self.config['num_steps'], **params)
       elif self.config['encoder'] == "SF":
           return SFEncoder(self.config['num_steps'], **params)
       elif self.config['encoder'] == "Deterministic":
           return DeterministicRate(self.config['num_steps'])
       else:
           raise ValueError(f"Coder no soportado")

   def init_decoder(self):
       decoder_type = self.config['decoder']
       params = self.config['decoder_params'].get(decoder_type, {})
       if decoder_type == "rate":
           return RateDecoder(self.config['num_steps'], **params)
       elif decoder_type == "latency":
           return LatencyDecoder(self.config['num_steps'], **params)
       elif decoder_type == "first_spike":
           return FirstSpikeDecoder(self.config['num_steps'], **params)
       elif decoder_type == "population_rate":
           return PopulationRateDecoder(self.config['num_steps'], **params)
       elif decoder_type == "rank_order":
           return RankOrderDecoder(self.config['num_steps'], **params)
       elif decoder_type == "all":
           return AllDecoders(self.config['num_steps'], **params)
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
       """
       Ejecuta el ciclo de entrenamiento y evaluación del experimento.
        - Entrena la red durante num_epochs epochs.
        - Evalúa la red cada eval_freq epochs.
        - Devuelve la precisión final obtenida por el decoder.
       """
       train_loader, test_loader = self.dataset.get_loaders()

       for epoch in range(self.config['num_epochs']):
           print(f"Epoch {epoch + 1}/{self.config['num_epochs']}")
           self.trainer.train_epoch(train_loader)

           if (epoch + 1) % self.config['eval_freq'] == 0:
               accuracy = self.trainer.evaluate(test_loader)

       print(f"\ndataset: {self.config['dataset']}. Encoder: {self.config['encoder']}. Architecture: {self.config['architecture']}. Decoder: {self.config['decoder']}")
       print(f"num_steps: {self.config['num_steps']}. Batch_size: {self.config['batch_size']}. Epochs: {self.config['num_epochs']}.")
       return accuracy

