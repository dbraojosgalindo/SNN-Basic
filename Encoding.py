import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import snntorch.spikegen as spikegen
import torch
import torch.nn.functional as F

class Encoder: #Clase padre de todas los encoders
    def __init__(self, num_steps):
        self.num_steps = num_steps

    def encode(self, data):
        raise NotImplementedError


class RateEncoder(Encoder):
    def __init__(self, num_steps, gain = 0.5):
        super().__init__(num_steps)
        self.gain = gain # reduce el % de frecuencia en ese gain %.
        
    def encode(self, data):
        return spikegen.rate(data, num_steps=self.num_steps, gain=self.gain)


class TtfsEncoder(Encoder):
    def __init__(self, num_steps, normalize=True, linear=False):
        super().__init__(num_steps)
        self.normalize = normalize # Distribuye los spikes en los num_steps
        self.linear = linear # Para el ecuacion de logaritmo interna
        
    def encode(self, data):
        return spikegen.latency(data, num_steps=self.num_steps, clip = True, normalize=self.normalize, linear=self.linear)
    
    
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
        output = torch.le(rand_data * self.rescale_fac, data).float()
        return output


    
class DeltaEncoder(Encoder):
    def __init__(self, num_steps, off_spike=False, threshold=0.1): # No se utiliza el num_steps
        super().__init__(num_steps)
        self.off_spike = off_spike
        self.threshold = threshold

    def encode(self, data): 
        #data.shape = torch.Size([128, 1, 28, 28])
        
        # 1. Permutar para que el Tiempo sea la Altura (Height)
        # La altura va a ser la dimensión temporal para el codificador Delta
        # Transformamos [Batch, Channel, Height, Width] -> [Height, Batch, Channel, Width]
        data_permuted = data.permute(2, 0, 1, 3)
        
        # 2. Aplicar Delta
        # padding=True mantiene el tamaño original (rellena la primera fila)
        spikes = spikegen.delta(data_permuted,
                                threshold=self.threshold,
                                padding=True,
                                off_spike=self.off_spike)
        
        # 3. Recuperar dimensiones originales
        spikes = spikes.permute(1, 2, 0, 3)
        
        # 4. Repetir temporalmente para la SNN
        return spikes.unsqueeze(0).repeat(self.num_steps, 1, 1, 1, 1)


class MWEncoder(Encoder): # Moving window
    # Spiking Neural Networks: Background, Recent Development and the NeuCube Architecture. https://github.com/KEDRI-AUT/snn-encoder-tools
    def __init__(self, num_steps, threshold=0.1, window = 5, off_spike=False):
        super().__init__(num_steps)
        self.threshold = threshold
        self.window = window
        self.off_spike = off_spike

    def encode(self, data) -> torch.Tensor:
        """
        Args:
            data (torch.Tensor): [Batch, Channel, Height, Width]
        Returns:
            torch.Tensor: [Num_Steps, Batch, Channel, Height, Width]
        """
        batch_size, channels, height, width = data.shape
        device = data.device
        
        kernel = torch.ones((channels, 1, 1, self.window), device=device) / self.window
        
        # 1. Padding a la izquierda (Replicate)
        padded_data = F.pad(data, (self.window, 0, 0, 0), mode='replicate')

        # 2. Convolución 2D Horizontal (Calcula medias)
        # groups=channels procesa cada canal independientemente
        means = F.conv2d(padded_data, kernel, groups=channels)
        
        # 3. Ajuste de tamaño (Recortar sobrante por padding)
        means = means[:, :, :, :-1]

        # 4. Comparación
        diff = data - means
        spikes = torch.zeros_like(data)
        spikes[diff > self.threshold] = 1.0
        if self.off_spike:
            spikes[diff < -self.threshold] = -1.0

        return spikes.unsqueeze(0).repeat(self.num_steps, 1, 1, 1, 1)
    

class SFEncoder(Encoder):
    
    # Spiking Neural Networks: Background, Recent Development and the NeuCube Architecture. https://github.com/KEDRI-AUT/snn-encoder-tools
    def __init__(self, num_steps = 25, threshold=0.1, off_spike=False):
        super().__init__(num_steps)
        self.threshold = threshold
        self.off_spike = off_spike

    def encode(self, data):
        # data: [Batch, Channel, Height, Width]
        batch_size, channels, height, width = data.shape
        device = data.device
        threshold_tensor = torch.tensor(self.threshold, device=device)
        
        outputs = torch.zeros_like(data)
        bases = torch.zeros_like(data)
        
        # Inicializar base con la primera columna (borde izquierdo)
        bases[:, :, :, 0] = data[:, :, :, 0]
        
        # Bucle sobre el ancho (Width) - Escáner Horizontal
        for w in range(1, width):
            current = data[:, :, :, w]
            prev_base = bases[:, :, :, w-1]
            
            # 1. Comprobar condiciones
            up = (current > prev_base + threshold_tensor)
            down = (current < prev_base - threshold_tensor)
            
            # 2. Generar Salida (Spikes)
            if self.off_spike:
                # Permite 1.0 y -1.0
                outputs[:, :, :, w] = torch.where(up, 1.0, torch.where(down, -1.0, 0.0))
            else:
                # Solo permite 1.0 (Ignora bajadas en la salida)
                outputs[:, :, :, w] = torch.where(up, 1.0, 0.0)
            
            # 3. Actualizar Base (Step)
            # NOTA: La base SIEMPRE se actualiza, incluso si off_spike=False.
            # Si no bajamos la base cuando la señal cae, no podremos detectar la próxima subida.
            delta_base = torch.where(up, threshold_tensor, torch.where(down, -threshold_tensor, 0.0))
            bases[:, :, :, w] = prev_base + delta_base

        # Expandir temporalmente
        return outputs.unsqueeze(0).expand(self.num_steps, -1, -1, -1, -1)