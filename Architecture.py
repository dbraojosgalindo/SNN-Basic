# Architecture.py
# Define las clases para construir arquitecturas de redes SNN.

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import torch.nn as nn
import snntorch as snn
import torch


class SNNArchitecture:
    def __init__(self, num_inputs, num_hidden, num_outputs, beta):
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs
        self.beta = beta

    def build_network(self):
        raise NotImplementedError


class TwoLayerSNN(SNNArchitecture):
    def build_network(self): 
        """
        Devuelve una two layer fully connected
        """
        class Net(nn.Module):
            def __init__(self, num_inputs, num_hidden, num_outputs, beta):
                super().__init__()
                # Primera capa totalmente conectada
                self.fc1 = nn.Linear(num_inputs, num_hidden)
                self.lif1 = snn.Leaky(beta=beta)  # Neuronas LIF para la primera capa
                # Segunda capa totalmente conectada
                self.fc2 = nn.Linear(num_hidden, num_outputs)
                self.lif2 = snn.Leaky(beta=beta)  # Neuronas LIF para la segunda capa

            def forward(self, x, num_steps):
                """
                Ejecuta la red durante num_steps pasos temporales.
                Devuelve los spikes y los potenciales de membrana de la segunda capa en cada paso.
                """
                mem1 = self.lif1.init_leaky()
                mem2 = self.lif2.init_leaky()

                spk2_rec = []
                mem2_rec = []

                for step in range(num_steps):
                    cur1 = self.fc1(x[step]) # entrada ponderada
                    spk1, mem1 = self.lif1(cur1, mem1) #spikes de la primera capa y su membrana
                    cur2 = self.fc2(spk1) # entrada ponderada de la segunda capa
                    spk2, mem2 = self.lif2(cur2, mem2) #spikes de la segunda capa y su membrana
                    spk2_rec.append(spk2)
                    mem2_rec.append(mem2)

                return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)

        return Net(self.num_inputs, self.num_hidden, self.num_outputs, self.beta)
