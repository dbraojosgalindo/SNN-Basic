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


class MWEncoder(Encoder): # Moving window
    # Spiking Neural Networks: Background, Recent Development and the NeuCube Architecture. https://github.com/KEDRI-AUT/snn-encoder-tools
    def __init__(self, num_steps, threshold=0.1, window = 5):
        super().__init__(num_steps)
        self.threshold = threshold
        self.window = window


    import torch

    def encode(self, data) -> torch.Tensor:
        """
        Codifica imágenes en secuencias de picos (+1/-1/0) usando moving window.
                
        Estrategia:
            Compara cada píxel con la media de una ventana temporal anterior (t - window - 1 a t - 1),
            generando picos cuando supera el umbral (±threshold respecto a la media).

        Parameters:
            data (torch.Tensor): Tensor de entrada en formato [batch, canales, alto, ancho]
            threshold (float): Umbral para activación de picos (absoluto, mismo rango que los datos)
            window (int): Tamaño de la ventana temporal histórica (en muestras/píxeles)

        Returns:
            torch.Tensor: Tensor codificado con valores {-1, 0, +1}. Mismo formato que la entrada.

        Algoritmo:
            1. Transformación a 1D:
                - Aplana las imágenes a vectores 1D (preserva batch y canales)
            2. Sumas acumuladas vectorizadas:
                - Calcula ventanas deslizantes mediante diferencias de sumas acumuladas
                - Padding inicial para manejar bordes
            3. Cálculo de medias:
                - Divide las sumas por el tamaño de ventana efectivo
                - Manejo especial para los primeros window+1 elementos (ventana inicial)
            4. Umbralización:
                - +1 si valor > media + threshold
                - -1 si valor < media - threshold
                - 0 en otro caso
        """
        batch_size, channels, height, width = data.shape
        data_flat = data.view(batch_size, -1)  # (batch_size, 784)
        seq_len = data_flat.size(1) # 784
        device = data.device
        
        # 1. Cálculo de sumas acumuladas con padding
        padded_cumsum = torch.cat([
            torch.zeros((batch_size, 1), device=device),
            torch.cumsum(data_flat, dim=1)
        ], dim=1)  # (batch_size, seq_len + 1)
        
        # 2. Índices y cálculos vectorizados
        t_indices = torch.arange(seq_len, device=device)  # (seq_len,)
        start_indices = t_indices - self.window - 1
        valid_start = torch.clamp(start_indices, min=0)  # (seq_len,)
        
        # 3. Obtener valores del padded_cumsum usando gather
        sum_window_start = torch.gather(
            padded_cumsum, 
            1, 
            valid_start.unsqueeze(0).expand(batch_size, -1).long()
        )
        
        sum_window_end = padded_cumsum[:, 1:seq_len+1]  # (batch_size, seq_len)
        sum_window_all = sum_window_end - sum_window_start
        
        # 4. Calcular sum_initial y máscaras
        sum_initial = padded_cumsum[:, self.window + 1] - padded_cumsum[:, 0]
        mask_initial = t_indices <= self.window
        
        # 5. Combinar sumas iniciales y móviles
        sum_total = torch.where(
            mask_initial.unsqueeze(0),
            sum_initial.unsqueeze(1).expand(-1, seq_len),
            sum_window_all
        )
        
        # 6. Calcular medias y aplicar umbral
        mean = sum_total / (self.window + 1)
        upper = data_flat > (mean + self.threshold)
        lower = data_flat < (mean - self.threshold)
        
        # 7. Generar salida codificada
        out = torch.zeros_like(data_flat, dtype=torch.float)
        out[upper] = 1
        out[lower] = -1
        
        return out.view(batch_size, channels, height, width).unsqueeze(0).repeat(self.num_steps, 1, 1, 1, 1)
    
    

class SFEncoder(Encoder):
    
    # Spiking Neural Networks: Background, Recent Development and the NeuCube Architecture. https://github.com/KEDRI-AUT/snn-encoder-tools
    def __init__(self, num_steps = 25, threshold=0.1):
        super().__init__(num_steps)
        self.threshold = threshold
    
    def encode(self, data):
        """
        Algoritmo de codificación Step-Forward (SF) para Redes Neuronales de Picos (SNNs).
    
        Transforma intensidades de píxeles en trenes temporales de picos (+1/-1/0) mediante 
        umbralización adaptativa. Compara cada píxel con una línea base dinámica, generando
        picos cuando se supera el umbral definido, en un esquema similar a modulación delta.

        Parameters:
            data (torch.Tensor): Batch de imágenes de entrada. Formato: [batch, canales, alto, ancho]
            
        Returns:
            torch.Tensor: Tensor de picos codificados. Formato: [num_steps, batch, canales, alto, ancho]
            
        Algoritmo:
            1. Inicialización: 
            - Establece línea base inicial (base) con el primer valor de píxel
            2. Comparación por Umbral:
            - Para cada píxel subsiguiente:
                * Pico = +1 si píxel > base + umbral (escalón positivo)
                * Pico = -1 si píxel < base - umbral (escalón negativo)
                * Sin pico (0) si está en [base ± umbral]
            3. Actualización de Base:
            - Ajusta la base ±umbral tras cada pico
        """
        batch_size, channels, height, width = data.shape
        images = data.view(batch_size, -1)  # [batch, 784]

        # Pre-asigna tensores en GPU si está disponible
        device = images.device
        threshold = torch.tensor(self.threshold, device=device)
        
        outputs = torch.zeros_like(images)
        bases = torch.zeros_like(images)
        bases[:, 0] = images[:, 0]

        # Vectorización del algoritmo SF para todo el batch
        for t in range(1, images.size(1)):
            current = images[:, t]
            prev_base = bases[:, t-1]
            
            # Cálculo vectorizado de condiciones
            up = (current > prev_base + threshold)
            down = (current < prev_base - threshold)
            
            # Actualización vectorizada
            outputs[:, t] = torch.where(up, 1.0, torch.where(down, -1.0, 0.0))
            bases[:, t] = prev_base + outputs[:, t] * threshold #Actualiza la base

        # Preparar salida para SNN (formato temporal)
        encoded = outputs.view(batch_size, channels, height, width)
        encoded = encoded.unsqueeze(0).expand(self.num_steps, -1, -1, -1, -1)
        
        return encoded