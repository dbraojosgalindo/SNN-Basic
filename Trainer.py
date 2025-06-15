# Trainer.py
# Define la clase Trainer, encargada del ciclo de entrenamiento y evaluación de la red SNN.

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import torch
from Encoding import Encoder
from Decoding import Decoder, AllDecoders
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
        """
        Realiza un epoch de entrenamiento sobre el conjunto de entrenamiento.
        - Codifica los datos de entrada con el encoder.
        - Ejecuta la red y acumula la pérdida a lo largo de los pasos temporales.
        - Actualiza los pesos de la red.
        """
        self.net.train()

        for data, targets in train_loader:
            # Codifica los datos de entrada
            data = self.encoder.encode(data).to(self.device)
            targets = targets.to(self.device)

            # Ejecuta la red SNN y obtiene los spikes y memorias
            spk_rec, mem_rec = self.net(data.view(self.num_steps, data.size(1), -1), self.num_steps)

            # Calcula la pérdida acumulada sobre todos los pasos temporales
            loss_val = torch.zeros((1), dtype=torch.float, device=self.device)
            for step in range(self.num_steps):
                loss_val += self.loss_fn(mem_rec[step], targets)

            self.optimizer.zero_grad()
            loss_val.backward()
            self.optimizer.step()

    def evaluate(self, test_loader):
        """
        Evalúa la red sobre el conjunto de test.
        - Codifica los datos de entrada con el encoder.
        - Ejecuta la red y decodifica la salida con el decoder.
        - Calcula la precisión final.
        """
        self.net.eval()
        total = 0
        correct = [0,0,0,0,0]  # Un contador para cada tipo de decoder
        accuracy = {}
        
        with torch.no_grad():
            for data, targets in test_loader:
                data = self.encoder.encode(data).to(self.device)
                targets = targets.to(self.device)

                spk_rec, _ = self.net(data.view(self.num_steps, data.size(1), -1), self.num_steps)
                decoded = self.decoder.decode(spk_rec)
                
                total += targets.size(0)
                # Si solo hay un decoder, decoded NO es una lista
                if not isinstance(decoded, list):
                    _, predicted = decoded.max(1)
                    correct[0] += (predicted == targets).sum().item()   
                else:
                    # Si hay múltiples decoders, iteramos sobre cada uno
                    for i in range(len(decoded)):
                        _, predicted = decoded[i].max(1)
                        correct[i] += (predicted == targets).sum().item()
                        
        # Si solo hay un decoder, devolvemos su precisión
        if not isinstance(decoded, list):
            accuracy[0] = 100 * correct[0] / total
            return accuracy[0]
        else:
            # Si hay múltiples decoders, calculamos la precisión para cada uno
            for i in range(len(decoded)):
                accuracy[i] = 100 * correct[i] / total
                #print(f"Test Set Accuracy for decoder {i}: {accuracy[i]:.2f}%")
            return accuracy