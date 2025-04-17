import snntorch.spikegen as spikegen


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