#include "trainer.hpp"
#include "network.hpp"
#include "imageloader.hpp"
#include "hyperparams.hpp"
#include <stdio.h>

float Trainer::outputErrorCache[10] = {0.0};
float Trainer::batchTrainingError = 0.0;

void Trainer::train(Network *network){
	imageloader::openFile();
	for(int batchCount=1; batchCount<= (HyperParams::numberOfInputs / HyperParams::batchSize);batchCount++){
		for(int count=0; count<HyperParams::batchSize; count++){
			network->cleanForNextInput();
			imageloader::loadNextImage();
			network->fetchInput();
			network->feedFoward();
			updateErrorCache(network);
		}

		calcAverageError();
		network->setError(outputErrorCache);
		calcAverageBatchError();
		logBatchError(batchCount);
		network->backProp();
		network->fixWeight();
		network->fixBias();
		cleanError();
	}

	imageloader::closeFile();
}

/*  updates the error cache AND training error*/
void Trainer::updateErrorCache(Network *network){
	float caseTrainingError  = 0.0;
	for(int i=0;i<10;i++){
		if(i == imageloader::getLabel()){
			// printf("i = %d, output = %f%%\n",i,network->getOutput(i));
			outputErrorCache[i] += network->getOutput(i) - 1;
			caseTrainingError += HyperParams::costFunction(network->getOutput(i), 1);
		}
		else{
			outputErrorCache[i] += network->getOutput(i) - 0;			
			caseTrainingError += HyperParams::costFunction(network->getOutput(i), 0);
		}
	}

	caseTrainingError /= 10;
	batchTrainingError += caseTrainingError;
}

void Trainer::calcAverageError(){
	for(int i=0; i<10; i++){
		outputErrorCache[i] /= HyperParams::batchSize;
	}
}

void Trainer::calcAverageBatchError(){
	batchTrainingError /= HyperParams::batchSize;
}

void Trainer::logBatchError(int batchCount){
	printf("Batch %d / error = %lf%%\n",batchCount,batchTrainingError);
}



void Trainer::cleanError(){
	for(int i=0; i<!0; i++)
		outputErrorCache[i] = 0;
	batchTrainingError = 0.0;
}