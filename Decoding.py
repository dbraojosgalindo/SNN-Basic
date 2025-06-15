# Decoding.py
# Define las clases de decodificadores para convertir la salida de spikes en predicciones.

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import torch
import snntorch.spikegen as spikegen


class Decoder:
   def __init__(self, num_steps, **kwargs):
       self.num_steps = num_steps


   def decode(self, spk_rec):
       raise NotImplementedError


class RateDecoder(Decoder):
   def __init__(self, num_steps, scale=1.0):
       super().__init__(num_steps)
       self.scale = scale


   def decode(self, spk_rec):
       return spk_rec.sum(dim=0) * self.scale

class LatencyDecoder(Decoder):
   def __init__(self, num_steps, target_time=0.5, sensitivity=1.0):
       super().__init__(num_steps)
       self.target_time = target_time
       self.sensitivity = sensitivity


   def decode(self, spk_rec):
       spike_times = torch.argmax(spk_rec, dim=0).float()
       normalized_times = spike_times / (self.num_steps - 1)
       time_diff = torch.abs(normalized_times - self.target_time)
       scores = torch.exp(-self.sensitivity * time_diff)
       return scores


class FirstSpikeDecoder(Decoder):
   def __init__(self, num_steps, threshold=0.1):
       super().__init__(num_steps)
       self.threshold = threshold


   def decode(self, spk_rec):
       spike_times = (spk_rec > self.threshold).float().argmax(dim=0)
       spike_times[(spk_rec <= self.threshold).all(dim=0)] = self.num_steps
       return torch.nn.functional.one_hot(
           spike_times.argmin(dim=-1),
           num_classes=spk_rec.shape[-1]
       ).float()


class PopulationRateDecoder(Decoder):
    def __init__(self, num_steps, num_classes=10, num_neurons_per_class=5):
        super().__init__(num_steps)
        self.num_classes = num_classes
        self.num_neurons_per_class = num_neurons_per_class

    def decode(self, spk_rec):
        population_activity = spk_rec.sum(dim=0)
        total_neurons = self.num_classes * self.num_neurons_per_class
        if population_activity.size(1) != total_neurons:
            raise ValueError(f"Expected {total_neurons} output neurons, got {population_activity.size(1)}")
        class_activity = population_activity.view(
            -1, self.num_classes, self.num_neurons_per_class
        ).sum(dim=2)
        return class_activity

class RankOrderDecoder(Decoder):
    def __init__(self, num_steps, num_classes):
        super().__init__(num_steps)
        self.num_classes = num_classes


    def decode(self, spk_rec):
        spike_times = (spk_rec > 0).float().argmax(dim=0)
        spike_times[(spk_rec > 0).float().sum(dim=0) == 0] = self.num_steps
        ranks = spike_times.argsort(dim=-1).argsort(dim=-1).float()
        scores = (self.num_classes - ranks) / self.num_classes
        return scores

class AllDecoders(Decoder):
    def __init__(self, num_steps, **kwargs):
        super().__init__(num_steps, **kwargs)
        self.rate = RateDecoder(num_steps)
        self.latency = LatencyDecoder(num_steps)
        self.firstSpike = FirstSpikeDecoder(num_steps)
        self.populationRate = PopulationRateDecoder(num_steps, 10, 1)
        self.rankOrder = RankOrderDecoder(num_steps, 10)
    
    def decode(self, spk_rec):
        outputs = []
        outputs.append(self.rate.decode(spk_rec))
        outputs.append(self.latency.decode(spk_rec))
        outputs.append(self.firstSpike.decode(spk_rec))
        outputs.append(self.populationRate.decode(spk_rec))
        outputs.append(self.rankOrder.decode(spk_rec))
        return outputs
    
    def get_nombre(i):
        """
        Devuelve el nombre del decodificador según el índice.
        """
        if i == 0:
            return "rate"
        if i == 1:
            return "latency"
        if i == 2:
            return "firstSpike"
        if i == 3:
            return "populationRate"
        if i == 4:
            return "rankOrder"