#include <math.h>
#include "hyperparams.hpp"

double HyperParams:: activate(double netValue){
	return 1/(1.0+exp(-netValue));
}

double HyperParams::dactivate(double netValue){
	return activate(netValue)*(1-activate(netValue));
}

double HyperParams::costFunction(double neuronOutput, double desiredOutput){
	return (0.5) * (neuronOutput - desiredOutput) * (neuronOutput - desiredOutput);
}
const double HyperParams::learningRate = 0.01;
const int HyperParams::batchSize = 15;
const int HyperParams::numberOfInputs = 5000;
const int HyperParams::hiddenLayerSize = 120;
const int HyperParams::repeatTimes = 30;
const int HyperParams::testSize = 1000;