#include "trainer.hpp"
#include "network.hpp"
#include "hyperparams.hpp"

#define IMAGE_WIDTH 28
#define IMAGE_HEIGHT 28
int main(){
	Network network(IMAGE_WIDTH*IMAGE_HEIGHT,HyperParams::hiddenLayerSize,10);
	network.initialize();
	Trainer::train(&network);
	return 0;
}