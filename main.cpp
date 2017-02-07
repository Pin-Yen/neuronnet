#include "trainer.hpp"
#include "network.hpp"
#include "hyperparams.hpp"
#include "imageloader.hpp"
#include "utils.hpp"
#include <time.h>
#define IMAGE_WIDTH 28
#define IMAGE_HEIGHT 28

int main(){
	time_t startTime, endTime;

	Network network(IMAGE_WIDTH*IMAGE_HEIGHT,HyperParams::hiddenLayerSize,10);
	network.initialize();
	imageloader::openFile();
	time(&startTime);
	Trainer::train(&network);
	time(&endTime);
	Utils::printElapsedTime(&startTime,&endTime);
	float accurarcy = Trainer::test(&network);
	Utils::writeLogFile(&endTime,accurarcy);
	imageloader::closeFile();
	return 0;
}