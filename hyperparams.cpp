#include <math.h>
#include "hyperparams.hpp"

float HyperParams:: activate(float netValue){
	return 1/(1.0+exp(-netValue));
}

float HyperParams::dactivate(float netValue){
	return activate(netValue)*(1-activate(netValue));
}

float HyperParams::costFunction(float neuronOutput, float desiredOutput){
	return (0.5) * (neuronOutput - desiredOutput) * (neuronOutput - desiredOutput);
}
const float HyperParams::learningRate = 0.01;
const int HyperParams::batchSize = 10;
const int HyperParams::numberOfInputs = 1000;
const int HyperParams::hiddenLayerSize = 300;
