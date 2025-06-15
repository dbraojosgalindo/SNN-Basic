import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import snntorch.spikegen as spikegen
import torch

class Encoder: 
    """
    Clase padre de todas los encoders
    """
    def __init__(self, num_steps):
        self.num_steps = num_steps

    def encode(self, data):
        raise NotImplementedError


class RateEncoder(Encoder):
    """
    Codifica los datos siguiendo una distribución de Bernoulli. Funciona igual que el PoissonEncoder.
    
    Args:
        gain: Escala la probabilidad de generacion de spikes
    """
    def __init__(self, num_steps, gain = 0.5):
        super().__init__(num_steps)
        self.gain = gain # reduce la frecuencia de spikes en ese gain %. 
        #Si un valor es 1 y el gain es 0.5 en vez de un 100% de spike tiene un 50%.
        print(self.gain)
        
    def encode(self, data):
        return spikegen.rate(data, num_steps=self.num_steps, gain=self.gain)


class TtfsEncoder(Encoder):
    """
    Codifica la data en el tiempo segun la intensidad del pixel.
    
    Args:
        rescale_fac: es como el gain pero inverso. Gain = 0.5 -> rescale = 2.0
    """
    def __init__(self, num_steps, normalize=True, linear=True):
        super().__init__(num_steps)
        self.normalize = normalize # Distribuye los spikes en los num_steps
        self.linear = linear 
        
    def encode(self, data):
        return spikegen.latency(data, num_steps=self.num_steps, clip = True, normalize=self.normalize, linear=self.linear)
    
    
class DirectEncoder(Encoder):
    """
    Sin codificacion ninguna
    """
    def __init__(self, num_steps):
        super().__init__(num_steps)
        
    def encode(self, data):
        # Se repiten num_steps veces la data
        return data.unsqueeze(0).repeat(self.num_steps, 1, 1, 1, 1) # -> torch.Size([num_steps, 128, 1, 28, 28])
    
class PoissonGen(Encoder):  #Rate vs Direct paper. No utiliza una distribucion de poisson, no se porque lo llaman asi en el paper
    """
    Genera spikes según una distribucion uniforme. Funciona igual que el RateEncoder.

    Args:
        rescale_fac (float): es como el gain pero inverso. Gain = 0.5 -> rescale = 2.0
    """
    def __init__(self, num_steps, rescale_fac=2.0): 
        super().__init__(num_steps)
        self.rescale_fac = rescale_fac
        
    def encode(self, data):
        #Genera unos numeros aleatorios entre [0-1) con size de la [num_steps, data] con una distribucion uniforme
        rand_data = torch.rand((self.num_steps,) + data.shape, device=data.device)

        # Se reescalan los valores random y se comparan con los de la data.
        output = torch.le(rand_data * self.rescale_fac, data).float()
        return output


    
class DeltaEncoder(Encoder):
    """
    Detecta cambios en las filas de cada imagen

    Args:
        off_spike (bool): Generar spikes negativos
    """
    def __init__(self, num_steps, off_spike=False): # No se utiliza el num_steps
        super().__init__(num_steps)
        self.off_spike = off_spike
        
    def encode(self, data): 
        #data.shape = torch.Size([128, 1, 28, 28])
        deltas = []
        for i in data: # Recorre cada imagen
            #Compara fila por fila de cada imagen sacando los spikes donde haya diferencia
            delta = spikegen.delta(i[0], off_spike=self.off_spike).unsqueeze(0).unsqueeze(0) 
            deltas.append(delta)
        solution = torch.cat(deltas, dim=0)
        
        #Duplica la solucion x los num_steps
        output = solution.unsqueeze(0).repeat(self.num_steps, 1, 1, 1, 1) # -> torch.Size([num_steps, 128, 1, 28, 28])
        return output


class MWEncoder(Encoder): # Moving window
    """
    Genera spikes usando una ventana movil. 
    
    Si la media del valor supera la media de la ventana anterior + un umbral se genera un spike

    Args:
        threshold (float): umbral para la activacion
        window (int): tamaño de la ventana 
    """
    # Spiking Neural Networks: Background, Recent Development and the NeuCube Architecture. https://github.com/KEDRI-AUT/snn-encoder-tools
    def __init__(self, num_steps, threshold=0.1, window = 5):
        super().__init__(num_steps)
        self.threshold = threshold
        self.window = window

    def encode(self, data) -> torch.Tensor:
        """
        Codifica imágenes en spikes usando moving window.
                
        Estrategia:
            Compara cada píxel con la media de una ventana temporal anterior (t - window - 1 a t - 1),
            generando picos cuando supera el umbral (±threshold respecto a la media).

        Parameters:
            data (torch.Tensor): Tensor de entrada en formato [batch, canales, alto, ancho]
            threshold (float): Umbral para activación de picos (absoluto, mismo rango que los datos)
            window (int): Tamaño de la ventana temporal histórica (en muestras/píxeles)

        Returns:
            torch.Tensor: Tensor codificado con valores {0, +1}. Formato [num_steps, batch, canales, alto, ancho]

        Algoritmo:
            1. Transformación a 1D:
                - Aplana las imágenes a vectores 1D (preserva batch y canales)
            2. Sumas acumuladas vectorizadas:
                - Calcula ventanas deslizantes mediante diferencias de sumas acumuladas
                - Padding inicial para manejar bordes
            3. Cálculo de medias:
                - Divide las sumas por el tamaño de ventana efectivo
                - Manejo especial para los primeros window+1 elementos (ventana inicial)
            4. Generar spikes:
                - +1 si valor > media + threshold
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
        
        # 7. Generar spikes
        out = torch.zeros_like(data_flat, dtype=torch.float)
        out[upper] = 1     
        
        return out.view(batch_size, channels, height, width).unsqueeze(0).repeat(self.num_steps, 1, 1, 1, 1)
    
    

class SFEncoder(Encoder):
    """
        Transforma intensidades de píxeles en spikes mediante un umbral adaptativa. 
        Compara cada píxel con una línea base dinámica + umbral absoluto, generando picos cuando se supera 
        el umbral definido.
    Args:
        threshold (_type_): Umbral absoluto

    """
    
    # Spiking Neural Networks: Background, Recent Development and the NeuCube Architecture. https://github.com/KEDRI-AUT/snn-encoder-tools
    def __init__(self, num_steps = 25, threshold=0.1):
        super().__init__(num_steps)
        self.threshold = threshold
    
    def encode(self, data):
        """
        Algoritmo de codificación Step-Forward (SF) para Redes Neuronales de Picos (SNNs).
    
        Transforma intensidades de píxeles en spikes  mediante un umbral adaptativa. 
        Compara cada píxel con una línea base dinámica, generando picos cuando se supera 
        el umbral definido.

        Parameters:
            data (torch.Tensor): Batch de imágenes de entrada. Formato: [batch, canales, alto, ancho]
            
        Returns:
            torch.Tensor: Tensor de picos codificados. Formato: [num_steps, batch, canales, alto, ancho]
            
        Algoritmo:
            1. Inicialización: 
            - Establece línea base inicial (base) con el primer valor de píxel
            2. Comparación por Umbral:
            - Para cada píxel:
                * Spike si píxel > base + umbral
                * Sin spike (0) si no
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
            outputs[:, t] = torch.where(up, 1.0, 0.0)
            delta_base = torch.where(up,  threshold, torch.where(down, -threshold, 0.0))
            bases[:, t] = prev_base + delta_base #Actualiza la base      

        # Preparar salida para SNN (formato temporal)
        encoded = outputs.view(batch_size, channels, height, width)
        encoded = encoded.unsqueeze(0).expand(self.num_steps, -1, -1, -1, -1)
        
        return encoded