import time
from Experiment import SNNExperiment

config = {
    'dataset': 'MNIST',
    'encoder': 'SF', # posibles: poisson, rate, ttfs, direct, ttfs_time, delta, MW, SF
    'decoder': 'rank_order', # posibles: rate, latency, first_spike, population,rank_order
    'architecture': 'TwoLayerSNN',
    'data_path': './data/mnist',
    'batch_size': 128,
    'num_inputs': 784,
    'num_hidden': 1000,
    'num_outputs': 10,
    'num_steps': 25,
    'beta': 0.95,
    'lr': 5e-4,
    'betas': (0.9, 0.999),
    'num_epochs': 1,
    'eval_freq': 1,
    'decoder_params': {
        'rate': {'scale': 1.0},
        'latency': {'target_time': 0.5, 'sensitivity': 1.0},
        'first_spike': {'threshold': 0.1},
        'population': {'num_classes': 10,'num_neurons_per_class': 1},
        'rank_order': {'num_classes': 10}
    }}

if __name__ == "__main__":
    start_time = time.time()
    experiment = SNNExperiment(config)
    final_accuracy = experiment.run()
    print(f"\n Final test accuracy: {final_accuracy:.2f}%")
    end_time = time.time()
    print(f"Tiempo de ejecución: {end_time - start_time:.4f} segundos")