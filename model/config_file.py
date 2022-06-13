# Transformer configuration
model_params = {'n_layers': 1,
                'n_heads': 8,
                'key_dim': 96}

# Training configuration
train_params = {'n_epochs': 100,
                'learning_rate': 0.001,
                'batch_size': 1}

# Data parameters
params = {'imagePath': './Datasets/',
          'dictFile': './Datasets/patient_dictionary.pickle',
          'dim': (384, 256),
          'timepoints': 32,
          'n_classes': 2}
