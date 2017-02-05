#include "layer.hpp"
#include "neuron.hpp"
#include "imageloader.hpp"
#define LAYERSIZE_MAX 28*28
#define IMAGE_WIDTH 28
#define IMAGE_HEIGHT 28

Layer::Layer(int size){
  layerSize = size;
  for(int i=0; i<layerSize;i++)
    neurons[i] = new Neuron();
}
Layer::~Layer(){
  for(int i=0; i<layerSize;i++)
    delete neurons[i];
}

void Layer::initialize(){
  for(int i=0;i<layerSize;i++){
    neurons[i]->initialize();
  }
}

void Layer::cleanForNextInput(){
  for(int i=0; i<layerSize; i++)
    neurons[i]->cleanForNextInput();
}

void Layer::fetchInput(){
  for(int r=0;r<IMAGE_HEIGHT;r++){
    for(int c=0; c<IMAGE_WIDTH; c++){
      neurons[r*IMAGE_HEIGHT+c]->setInput(imageloader::getPixel(r,c));
    }
  }
}

void Layer::feedFoward(){
  for(int i=0;i<layerSize;i++){
    neurons[i]->feedFoward();
  }
}

void Layer::backProp(){
  for(int i=0;i<layerSize;i++){
    neurons[i]->backProp();
  }
}

void Layer::connectPreviousLayer(Layer *previousLayer){
  for(int i=0;i<layerSize;i++){
    neurons[i]->connectPreviousLayer(previousLayer);
  }
}

void Layer::connectNextLayer(Layer *nextlayer){
  for(int i=0;i<layerSize;i++){
    neurons[i]->connectNextLayer(nextlayer);
  }
}

void Layer::setError(float errorArr[10]){
  for(int i=0; i<10; i++){
    neurons[i]->setDeltaNext(errorArr[i]);
  }
}

void Layer::fixWeight(){
  for(int i=0;i<layerSize;i++){
    neurons[i]->fixWeight();
  }
}

void Layer::fixBias(){
  for(int i=0;i<layerSize;i++){
    neurons[i]->fixBias();
  }
}

Neuron* Layer::getNeuronPtr(int index){
  return neurons[index];
}

int Layer::getLayerSize(){
  return layerSize;
}
