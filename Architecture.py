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
        class Net(nn.Module):
            def __init__(self, num_inputs, num_hidden, num_outputs, beta):
                super().__init__()
                self.fc1 = nn.Linear(num_inputs, num_hidden)
                self.lif1 = snn.Leaky(beta=beta)
                self.fc2 = nn.Linear(num_hidden, num_outputs)
                self.lif2 = snn.Leaky(beta=beta)

            def forward(self, x, num_steps):
                mem1 = self.lif1.init_leaky()
                mem2 = self.lif2.init_leaky()

                spk2_rec = []
                mem2_rec = []

                for step in range(num_steps):
                    cur1 = self.fc1(x[step])
                    spk1, mem1 = self.lif1(cur1, mem1)
                    cur2 = self.fc2(spk1)
                    spk2, mem2 = self.lif2(cur2, mem2)
                    spk2_rec.append(spk2)
                    mem2_rec.append(mem2)

                return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)

        return Net(self.num_inputs, self.num_hidden, self.num_outputs, self.beta)
