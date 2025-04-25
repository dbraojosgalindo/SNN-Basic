import snntorch.spikegen as spikegen
import torch
import numpy as np

class Encoder:
    def __init__(self, num_steps):
        self.num_steps = num_steps

    def encode(self, data):
        raise NotImplementedError


class RateEncoder(Encoder):
    def __init__(self, num_steps, gain = 0.5):
        super().__init__(num_steps)
        self.gain = gain
        
    def encode(self, data):
        return spikegen.rate(data, num_steps=self.num_steps, gain=self.gain)


class TtfsEncoder(Encoder):
    def __init__(self, num_steps, normalize=True, linear=True):
        super().__init__(num_steps)
        self.normalize = normalize
        self.linear = linear
        
    def encode(self, data):
        return spikegen.latency(data, num_steps=self.num_steps, clip = True, normalize=self.normalize, linear=self.linear)
    
class Ttfs_time_Encoder(Encoder): # Transforma la data em tiempos de spikes no en spikes
    def __init__(self, num_steps, p = 0.0, q = 1.0):
        super().__init__(num_steps)
        self.q, self.p = q, p
        
    def encode(self, data): # High-performance deep spiking neural networks with 0.3 spikes per neuron
        """
        Convert input values into time-to-first-spike spiking times.
        """
        #self.x_test, self.x_train = (self.x_test - self.p)/(self.q-self.p), (self.x_train - self.p)/(self.q-self.p)
        self.data = (data - self.p)/(self.q-self.p)
        #self.x_train, self.x_test=1 - np.array(self.x_train), 1 - np.array(self.x_test)
        self.data = 1 - np.array(self.data)
        self.data = torch.from_numpy(self.data)
        return self.data.unsqueeze(0).repeat(self.num_steps, 1, 1, 1, 1) # -> torch.Size([num_steps, 128, 1, 28, 28])
    
    
class DirectEncoder(Encoder):
    def __init__(self, num_steps):
        super().__init__(num_steps)
        
    def encode(self, data):
        return data.unsqueeze(0).repeat(self.num_steps, 1, 1, 1, 1) # -> torch.Size([num_steps, 128, 1, 28, 28])
    
class PoissonGen(Encoder):  #Rate vs Direct paper
    def __init__(self, num_steps, rescale_fac=2.0): #rescale es como el gain pero inverso. Gain = 0.5 -> rescale = 2.0
        super().__init__(num_steps)
        self.rescale_fac = rescale_fac
        
    def encode(self, data):
        rand_data = torch.rand((self.num_steps,) + data.shape, device=data.device)

        # Crear máscara y multiplicar por el signo
        mask = torch.le(rand_data * self.rescale_fac, torch.abs(data)).float()
        output = mask * torch.sign(data)
        return output


    
class DeltaEncoder(Encoder):
    def __init__(self, num_steps, off_spike=True): # No se utiliza el num_steps
        super().__init__(num_steps)
        self.off_spike = off_spike
        
    def encode(self, data): 
        #data.shape = torch.Size([128, 1, 28, 28])
        deltas = []
        for i in data:
            delta = spikegen.delta(i[0], off_spike=self.off_spike).unsqueeze(0).unsqueeze(0) #Si se usa con imagenes en rgb hay que cambiar esto
            deltas.append(delta)
        solution = torch.cat(deltas, dim=0)
        #Duplica la solucion x los num_steps
        final = solution.unsqueeze(0).repeat(self.num_steps, 1, 1, 1, 1) # -> torch.Size([num_steps, 128, 1, 28, 28])
        return final
