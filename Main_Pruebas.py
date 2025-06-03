import time
import csv
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
from Experiment import SNNExperiment
from Decoding import AllDecoders

num_steps = [5, 10, 25, 50, 100, 200]
decoders = ['all']

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
    'eval_freq': 32,
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
    output = {}
    for i in range(len(accuracy)):
        decoder_i = AllDecoders.get_nombre(i)
        accuracy_i = accuracy[i]
        print(f"decoder={decoder_i}, num_steps={num_step}, trial={trial} → Final accuracy: {accuracy_i:.2f}%")
        output[i] = {
            'encoder': config['encoder'],
            'decoder': decoder_i,
            'num_steps': num_step,
            'accuracy': f"{accuracy_i:.2f}",
            'numero de prueba': trial
        }
    return output

    
if __name__ == "__main__":
    start_time = time.time()
    all_futures = []

    with ProcessPoolExecutor(max_workers=10) as executor:
        for decoder in decoders:
            for num_step in num_steps:
                for trial in range(1, 6):
                    future = executor.submit(run_experiment, decoder, num_step, trial, config_base)
                    all_futures.append(future)

        results = [future.result() for future in as_completed(all_futures)]

    with open(f'results {config_base["encoder"]}.csv', 'w', newline='') as csvfile:
        fieldnames = ['encoder', 'decoder', 'num_steps', 'accuracy', 'numero de prueba']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            for i in row:
                writer.writerow(row[i])

        total_time = time.time() - start_time
        csvfile.write(f"# Tiempo total de ejecucion: {total_time:.4f} segundos\n")

    print(f"Tiempo de ejecución: {total_time:.4f} segundos")