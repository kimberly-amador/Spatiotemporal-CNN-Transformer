import pickle
import metrics
import tensorflow.keras.backend as K
import config_file as cfg
from modules import ENCODER, DECODER, TRANSFORMER
from data_generator import DataGenerator2D
from lr_scheduler import StepDecay
from tensorflow.keras import optimizers, Model
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.layers import Concatenate, Input, Lambda


# -----------------------------------
#            PARAMETERS
# -----------------------------------

# Print config parameters
print('TRANSFORMER PARAMETERS')
for x in cfg.model_params:
    print(x, ':', cfg.model_params[x])
print('\nTRAINING PARAMETERS')
for x in cfg.train_params:
    print(x, ':', cfg.train_params[x])
print('\nDATA INFORMATION')
for x in cfg.params:
    print(x, ':', cfg.params[x])

# -----------------------------------
#            IMPORT DATA
# -----------------------------------

# Importing patient dictionary
print("[INFO] loading patient dictionary...")
with open(cfg.params['dictFile'], 'rb') as output:
    partition = pickle.load(output)

# Calling training generator
train_generator = DataGenerator2D(partition['train'], shuffle=True, **cfg.params, **cfg.train_params)
val_generator = DataGenerator2D(partition['val'], shuffle=False, **cfg.params, **cfg.train_params)

# -----------------------------------
#            BUILD MODEL
# -----------------------------------

# Define metrics and loss
dice_loss = metrics.Dice(nb_labels=cfg.params['n_classes']).loss
mean_dice = metrics.Dice(nb_labels=cfg.params['n_classes']).mean_dice

print("[INFO] building model...")
base_network = ENCODER.build(shape=cfg.params['dim'])
absolute_diff = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))  # compute absolute difference between tensors

# Initialize encoder
inputs = []
outputs = []
skip = []
for w in range(cfg.params['timepoints']):
    # Create inputs and get model outputs
    i = Input(shape=(*cfg.params['dim'], 1))
    o = base_network(i)
    # Append results
    inputs.append(i)
    outputs.append(o[0])
    skip.append(o[1:])

# Concatenate latent vectors and skip connections
encoders = Concatenate(axis=1)(outputs)
connections = []
for i in range(len(skip[0])):
    subtract = []
    for j in range(cfg.params['timepoints']):
        subtract = skip[j][i] if j == 0 else absolute_diff([subtract, skip[j][i]])
    connections.append(subtract)

# Merge temporal information via Transformer
tmp = TRANSFORMER.build(encoders, timepoints=cfg.params['timepoints'], **cfg.model_params)

# Get decoder output
CNN_TCN_output = DECODER.build(inputTensor=tmp, down=connections, n_classes=cfg.params['n_classes'])

# Create model and plot its summary
CNN_TCN_model = Model(inputs, CNN_TCN_output)
CNN_TCN_model.summary()

# Compile model
CNN_TCN_model.compile(loss=dice_loss, optimizer=optimizers.Adam(), metrics=[mean_dice])

# -----------------------------------
#            TRAIN MODEL
# -----------------------------------

# Callbacks
sch = StepDecay(initAlpha=cfg.train_params['learning_rate'], factor=0.25, dropEvery=15)
callbacks = [LearningRateScheduler(sch)]

# Train model
print("[INFO] training network for {} epochs...".format(cfg.train_params['n_epochs']))
H = CNN_TCN_model.fit(x=train_generator,
                      validation_data=val_generator,
                      epochs=cfg.train_params['n_epochs'],
                      callbacks=callbacks)

# Save model
CNN_TCN_model.save_weights('model_weights')
print("[INFO] model saved to disk.")

# Save training history
# np.save('results/trainHistoryDic.npy', H.history)
