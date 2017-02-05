#include "network.hpp"
#include "neuron.hpp"
#define IMAGE_WIDTH 28
#define IMAGE HEIGHT 28

	/*constructor for 3 layer network*/
Network::Network(int inputLayerSize, int hiddenLayerSize, int outputLayerSize){
	inputLayer = new Layer(inputLayerSize);
	// inputLayer = Layer(inputLayerSize);

	hiddenLayer = new Layer(hiddenLayerSize);
	// hiddenLayer = Layer(hiddenLayerSize);
	hiddenLayer->connectPreviousLayer(inputLayer);
	inputLayer->connectNextLayer(hiddenLayer);

	outputLayer = new Layer(outputLayerSize);
	// outputLayer = Layer(outputLayerSize);
	outputLayer->connectPreviousLayer(hiddenLayer);
	hiddenLayer->connectNextLayer(outputLayer);
}
Network::~Network(){
	delete inputLayer;
	delete hiddenLayer;
	delete outputLayer;
}

void Network::initialize(){
	inputLayer->initialize();
	hiddenLayer->initialize();
	outputLayer->initialize();
}

void Network::cleanForNextInput(){
	inputLayer->cleanForNextInput();
	hiddenLayer->cleanForNextInput();
	outputLayer->cleanForNextInput();
}

void Network::cleanForNextBatch(){
	inputLayer->cleanForNextBatch();
	hiddenLayer->cleanForNextBatch();
	outputLayer->cleanForNextBatch();	
}
void Network::fetchInput(){
	inputLayer->fetchInput();
}

void Network::feedFoward(){
	inputLayer->feedFoward();
	hiddenLayer->feedFoward();
	outputLayer->feedFoward();
}

void Network::backProp(){
	outputLayer->backProp();
	hiddenLayer->backProp();
	inputLayer->backProp();
}

void Network::fixWeight(){
	outputLayer->fixWeight();
	hiddenLayer->fixWeight();
	inputLayer->fixWeight();
}

void Network::fixBias(){
	outputLayer->fixBias();
	hiddenLayer->fixBias();
	inputLayer->fixBias();
}

float Network::getOutput(int index){
	return outputLayer->getNeuronPtr(index)->getOutput();
}

void Network::setError(float errorArr[10]){
	outputLayer->setError(errorArr);
}
