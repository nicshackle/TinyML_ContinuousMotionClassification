## Continuous motion classification with TinyML (Tensorflow Lite)

A simple example of continuous motion classification with Tiny ML. There are a number of one-shot gesture classification examples, but at the time of writing I didn't find many continuous motion examples (not using Impulse Edge).

### Hardware
This example uses the Arduino Sense BLE 33

### Usage
1) Upload the motion capture sketch to the Arduino
2) Record motion data and save as a CSV
3) Edit the python script to include your CSV file names
4) Edit the python script to normalise your data appropriately
5) Train model
6) Convert model to C
7) Add model to model.h
8) Edit the Arduino sketch to load tensors in the same way the the python script does (see notes in code)
9) Classify!


