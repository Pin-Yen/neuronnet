#include <stdlib.h>
#include <time.h>
#include "utils.hpp"

bool Utils::randSeedInitialized = false;

void Utils::randSetSeed(){
  srand(time(NULL));
  randSeedInitialized = true;
}

float Utils::randFloat(){
	if(!randSeedInitialized)
		randSetSeed();

  int i = rand()%10000;
  return (i / 10000.0) -0.5;
}