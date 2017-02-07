#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include "utils.hpp"
#include "hyperparams.hpp"
#define LOGFILE_NAME "log.txt"
bool Utils::randSeedInitialized = false;

void Utils::randSetSeed(){
  srand(time(NULL));
  randSeedInitialized = true;
}

void Utils::printElapsedTime(time_t *start, time_t *end){
	printf("Training time : %lf seconds\n",difftime(*end,*start));
}
void Utils::writeLogFile(time_t *endTime ,float accurarcy){
	FILE *logFile = fopen(LOGFILE_NAME,"a");

	fprintf(logFile,"ac: %3.1f%% / ", accurarcy*100);
	fprintf(logFile,"sampleSize: %4d / ",HyperParams::numberOfInputs);
	fprintf(logFile,"learnrate: %5.5f / ",HyperParams::learningRate);
	fprintf(logFile,"batsize: %4d / " ,HyperParams::batchSize);
	fprintf(logFile, "batrep: %3d / ",HyperParams::repeatTimes );
	fprintf(logFile,"hid: %4d / ", HyperParams::hiddenLayerSize);
	char timeStr[50];
	strftime(timeStr,50,"%F",localtime(endTime));
	fprintf(logFile,"%s\n",timeStr);
	fclose(logFile);
}

double Utils::randdouble(){
	if(!randSeedInitialized)
		randSetSeed();

  int i = rand()%10000000;
  return (i / 10000000.0) -0.5;
}