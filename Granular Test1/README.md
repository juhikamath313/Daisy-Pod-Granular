# test

## Author

Juhi Kamath

## Description

Granular synth patch for the electrosmith daisy pod. Allows for recording up to a 15 second stereo audio clip for granulation. Includes DSP for normalization & reverb. UI is made of 3 states, swapped between by clicking the encoder. 

1: pot 1 = grain start; pot 2 = grain length; 
2: pot 1 = grain density; pot 2 = grain randomized start position;
3: pot 1 = reverb dry/wet; pot 2 = reverb decay;

note can be changed by rotating the encoder. Clip record with button 1 and playback with button 2. When not recording or playing, patch will act as a passthrough.

PCB & standalone circuitry to work with the daisy seed is in development.

flash in VSCode using task: build & program DFU
