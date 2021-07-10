import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

# the list of motions that data is available for
MOTIONS = [
    "walking",
    "still",
]

SAMPLES_PER_MOTION = 119

NUM_MOTIONS = len(MOTIONS)

# create a one-hot encoded matrix that is used in the output
ONE_HOT_ENCODED_MOTIONS = np.eye(NUM_MOTIONS)

inputs = []
outputs = []

# read each csv file and push an input and output
for motion_index in range(NUM_MOTIONS):
  motion = MOTIONS[motion_index]
  print(f"Processing index {motion_index} for motion '{motion}'.")
  
  output = ONE_HOT_ENCODED_MOTIONS[motion_index]
  
  df = pd.read_csv(motion + ".csv")
  
  # calculate the number of motion recordings in the file
  num_recordings = int(df.shape[0] / SAMPLES_PER_MOTION)
  
  print(f"\tThere are {num_recordings} recordings of the {motion} motion.")
  
  for i in range(num_recordings):
    tensor = []
    for j in range(SAMPLES_PER_MOTION):
      index = i * SAMPLES_PER_MOTION + j
      # inspect graphs of the data, and 
      # normalize the input data to fit between 0 to 1.
      # Max ranges are:
      # - acceleration between: -4 to +4 (i.e. (float(df['aX'][index]) + 4) / 8)
      # - gyroscope between: -2000 to +2000 (i.e. (float(df['gX'][index]) + 2000) / 4000)
      # However, in cases where you don't want to use the full range:
      # Example: (float(df['aY'][index]) / 0.8)
      #
      # Note that in this example, we just use one axis. 
      tensor += [
          (float(df['aY'][index]) / 0.8)
      ]

    inputs.append(tensor)
    outputs.append(output)

# convert the list to numpy array
inputs = np.array(inputs)
outputs = np.array(outputs)

print("Data set parsing and preparation complete.")

# Randomize the order of the inputs, so they can be evenly distributed for training, testing, and validation
# https://stackoverflow.com/a/37710486/2020087
num_inputs = len(inputs)
randomize = np.arange(num_inputs)
np.random.shuffle(randomize)

# Swap the consecutive indexes (0, 1, 2, etc) with the randomized indexes
inputs = inputs[randomize]
outputs = outputs[randomize]

# Split the recordings (group of samples) into three sets: training, testing and validation
TRAIN_SPLIT = int(0.6 * num_inputs)
TEST_SPLIT = int(0.2 * num_inputs + TRAIN_SPLIT)

inputs_train, inputs_test, inputs_validate = np.split(inputs, [TRAIN_SPLIT, TEST_SPLIT])
outputs_train, outputs_test, outputs_validate = np.split(outputs, [TRAIN_SPLIT, TEST_SPLIT])

print("Data set randomization and splitting complete.")

# build the model and train it
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(50, activation='relu')) # relu is used for performance
model.add(tf.keras.layers.Dense(15, activation='relu'))
model.add(tf.keras.layers.Dense(NUM_MOTIONS, activation='softmax')) # softmax is used, because we only expect one motion to occur per input
model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
history = model.fit(inputs_train, outputs_train, epochs=600, batch_size=1, validation_data=(inputs_validate, outputs_validate))

# increase the size of the graphs. The default size is (6,4).
plt.rcParams["figure.figsize"] = (20,10)

# graph the loss, the model above is configure to use "mean squared error" as the loss function
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'g.', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

print(plt.rcParams["figure.figsize"])

# graph the loss again skipping a bit of the start
SKIP = 100
plt.plot(epochs[SKIP:], loss[SKIP:], 'g.', label='Training loss')
plt.plot(epochs[SKIP:], val_loss[SKIP:], 'b.', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# graph of mean absolute error
mae = history.history['mae']
val_mae = history.history['val_mae']
plt.plot(epochs[SKIP:], mae[SKIP:], 'g.', label='Training MAE')
plt.plot(epochs[SKIP:], val_mae[SKIP:], 'b.', label='Validation MAE')
plt.title('Training and validation mean absolute error')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.legend()
plt.show()

# use the model to predict the test inputs
predictions = model.predict(inputs_test)

# print the predictions and the expected ouputs
print("predictions =\n", predictions)
print("actual =\n", outputs_test)

# Convert the model to the TensorFlow Lite format without quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model to disk
open("motion_model.tflite", "wb").write(tflite_model)
  
import os
basic_model_size = int(os.path.getsize("motion_model.tflite"))/1000
print("Model is "+str(basic_model_size)+"Kb")
print("convert model with: cat motion_model.tflite | xxd -i >> model.h")
