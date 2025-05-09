import time
import csv
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Pool
from Experiment import SNNExperiment

num_steps = [10, 25, 50, 100, 200]
decoders = ['rate', 'latency', 'first_spike', 'rank_order']

# Config base (fuera del proceso paralelo para seguridad)
config_base = {
    'dataset': 'MNIST',
    'encoder': 'rate',
    'decoder': '',
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
    'num_epochs': 32,
    'eval_freq': 1,
    'decoder_params': {
        'rate': {'scale': 1.0},
        'latency': {'target_time': 0.5, 'sensitivity': 1.0},
        'first_spike': {'threshold': 0.1},
        'population': {'num_classes': 10, 'num_neurons_per_class': 1},
        'rank_order': {'num_classes': 10}
    }
}

def run_experiment(decoder, num_step, trial, base_config):
    config = copy.deepcopy(base_config)
    config['decoder'] = decoder
    config['num_steps'] = num_step
    experiment = SNNExperiment(config)
    accuracy = experiment.run()
    print(f"decoder={decoder}, num_steps={num_step}, trial={trial} → Final accuracy: {accuracy:.2f}%")
    return {
        'encoder': config['encoder'],
        'decoder': decoder,
        'num_steps': num_step,
        'accuracy': f"{accuracy:.2f}",
        'numero de prueba': trial
    }

if __name__ == "__main__":
    start_time = time.time()
    all_futures = []

    n_processes = 10
    pool = Pool(n_processes)

    async_results = []
    results = []

    for decoder in decoders:
        for num_step in num_steps:
            for trial in range(1, 6):
                async_result = pool.apply_async(run_experiment, args=(decoder, num_step, trial, config_base))
                async_results.append((decoder, num_step, trial, async_result))

    for decoder, num_step, trial, async_result in async_results:
        result = async_result.get()
        results.append(result)

    with open(f'results {config_base["encoder"]}.csv', 'w', newline='') as f:
        fieldnames = ['encoder', 'decoder', 'num_steps', 'accuracy', 'numero de prueba']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

        total_time = time.time() - start_time
        f.write(f"# Tiempo total de ejecucion: {total_time:.4f} segundos\n")

    print(f"Tiempo de ejecución: {total_time:.4f} segundos")
