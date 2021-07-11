## Continuous motion classification with TinyML (Tensorflow Lite)

A simple example of continuous motion classification with Tiny ML. There are a number of one-shot gesture classification examples, but at the time of writing I didn't find many continuous motion examples (not using Impulse Edge).

[![video](https://github.com/nicshackle/TinyML_ContinuousMotionClassification/blob/main/docs/video.png?raw=true)](https://vimeo.com/manage/videos/573416024)

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

### Data normalization

In order for the model to classify motions correctly, attention needs to be paid to how you normalise the motion data. 
Consider the below sample (±1 sec) of motion data for _walking_:
![Walking motion data](https://github.com/nicshackle/TinyML_ContinuousMotionClassification/blob/main/docs/oneSample.png?raw=trueg)

In this project, only the Y axis was used for training and classification (see pink bounding box).

Given that the data is between the values of 0.0 and 1.1, a normalisation of `y*0.8` was chosen such that values fit between 0-1 (required for passing into a tensor).

I found that a) using the full range of the IMU b) using all data from the IMU (gyro+accel) negatively impacted accuracy. 


