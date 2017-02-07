#include "trainer.hpp"
#include "network.hpp"
#include "imageloader.hpp"
#include "hyperparams.hpp"
#include <stdio.h>

double Trainer::outputErrorCache[10] = {0.0};
double Trainer::batchTrainingError = 0.0;
int Trainer::rightCount = 0;

void Trainer::train(Network *network){
	logBasicInfo();

	for(int batchCount=1; batchCount<= (HyperParams::numberOfInputs / HyperParams::batchSize);batchCount++){

		for(int repeatCount =0; repeatCount < HyperParams::repeatTimes;repeatCount++){
			network->cleanForNextBatch();
			cleanBatchError();
			
			for(int count=0; count<HyperParams::batchSize; count++){
				network->cleanForNextInput();
				imageloader::loadNextImage();
				network->feedFoward();
				// if(batchCount%10 == 0 && repeatCount == 0)
					// printPredictDetail(network);
				updateErrorCache(network);
				network->setError(outputErrorCache);
				network->backProp();
			}
			network->fix();
			if(repeatCount == 0)
				logBatchError(batchCount);

			if(repeatCount != HyperParams::repeatTimes-1){
				imageloader::rewind(HyperParams::batchSize);
			}
		}
	}

}

float Trainer::test(Network *network){
	int rightCount = 0;
	for(int i=0;i<HyperParams::testSize;i++){
		imageloader::loadNextImage();
		network->cleanForNextInput();
		network->feedFoward();
		if(((int)imageloader::getLabel()) == network->getAnswer())
			rightCount++;
	}
	logBasicInfo();
	printf("Accurarcy = %f\n",((float)rightCount)/HyperParams::testSize);
	return ((float)rightCount)/HyperParams::testSize;
}

/*  updates the error cache AND training error*/
void Trainer::updateErrorCache(Network *network){
	double caseTrainingError  = 0.0;
	for(int i=0;i<10;i++){
		if(i == imageloader::getLabel()){
			outputErrorCache[i] = network->getOutput(i) - 1;
			caseTrainingError += HyperParams::costFunction(network->getOutput(i), 1);
		}
		else{
			outputErrorCache[i] = network->getOutput(i) - 0;			
			caseTrainingError += HyperParams::costFunction(network->getOutput(i), 0);
		}
	}
	caseTrainingError /= 10;
	batchTrainingError += caseTrainingError;

	if(network->getAnswer() == (int)imageloader::getLabel()){
		rightCount ++;
	}
}



void Trainer::logBatchError(int batchCount){
	batchTrainingError /= HyperParams::batchSize;
	printf("Batch %d / Error = %lf%%",batchCount,batchTrainingError);
	printf(" / Accurarcy = %lf%%\n",(double)rightCount/HyperParams::batchSize);
}


/* resets batchError and rightCount */
void Trainer::cleanBatchError(){
	batchTrainingError = 0.0;
	rightCount = 0;
}

void Trainer::logBasicInfo(){
	printf("Learning Rate : %lf\n",HyperParams::learningRate);
	printf("Batch Size %d\n",HyperParams::batchSize);
	printf("Sample Amount: %d\n",HyperParams::numberOfInputs);
	printf("Batch Repeat: %d\n",HyperParams::repeatTimes);
	printf("--------------------------------------------\n");
}

void Trainer::printPredictDetail(Network *network){
	printf("label : %d\n",(int)imageloader::getLabel());
	for(int i=0;i<10;i++){
		printf("%d: %lf%%\n",i,network->getOutput(i));
	}
}

