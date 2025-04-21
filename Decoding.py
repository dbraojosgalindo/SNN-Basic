import torch
import snntorch.spikegen as spikegen


class Decoder:
    def __init__(self, num_steps):
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
    def decode(self, spk_rec):
        significant_spikes = (spk_rec > self.threshold).float()
        spike_times = torch.argmax(significant_spikes, dim=0)
        no_spike_mask = (significant_spikes.sum(dim=0) == 0)
        spike_times[no_spike_mask] = self.num_steps
        winning_neurons = torch.argmin(spike_times, dim=1)
        output = torch.zeros_like(spk_rec[0])
        output[torch.arange(output.size(0)), winning_neurons] = 1

        return output