from Experiment import SNNExperiment

config = {
    'dataset': 'MNIST',
    'encoder': 'rate',
    'decoder': 'rate',
    'architecture': 'TwoLayerSNN',
    'data_path': '/tmp/data/mnist',
    'batch_size': 128,
    'num_inputs': 784,
    'num_hidden': 1000,
    'num_outputs': 10,
    'num_steps': 25,
    'beta': 0.95,
    'lr': 5e-4,
    'betas': (0.9, 0.999),
    'num_epochs': 1,
    'eval_freq': 1
}

experiment = SNNExperiment(config)
final_accuracy = experiment.run()
