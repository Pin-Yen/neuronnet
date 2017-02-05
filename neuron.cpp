#include "neuron.hpp"
#include "layer.hpp"
#include "utils.hpp"
#include "hyperparams.hpp"

  /*constructor*/
Neuron::Neuron(){
  netValue = 0;
  output = 0;
  for(int i=0; i<LAYERSIZE_MAX;i++)
    weightError[i] = 0;
  biasError = 0;
}

  /* initialize weight and bias to a random value*/
void Neuron::initialize(){
    /*initialize weight*/
  for(int i=0;i<previousLayerSize;i++)
    weight[i] = Utils::randFloat();

    /*initialize bias*/
  bias = Utils::randFloat();
}

void Neuron::connectPreviousLayer(Layer *previousLayer){
  previousLayerSize = previousLayer->getLayerSize();
  for(int i=0;i<previousLayerSize;i++){
    previousLayerNeurons[i] = previousLayer->getNeuronPtr(i);
  }
}

void Neuron::connectNextLayer(Layer *nextLayer){
  nextLayerSize = nextLayer->getLayerSize();
  for(int i=0;i<nextLayerSize;i++){
    nextLayerNeurons[i] = nextLayer->getNeuronPtr(i);
  }
}

  /* ONLY BE CALLED WHEN RESETTING THE WHOLE TRAINING PROCESS
   * sets All variables to 0, except those determined by the network structure (i.e. nextLayer, previousLayer)
   * not to be confused with clearForNextInput */
void Neuron::resetAll(){
  netValue = 0;
  output = 0;
  bias = 0;
  delta = 0;
  deltaNext = 0;
  for(int i=0;i<LAYERSIZE_MAX;i++){
    weight[i] = 0;
    weightError[i] = 0;
  }
}

void Neuron::feedFoward(){
  calcNet();
  calcOutput();
}

void Neuron::backProp(){
  calcDelta();
  backPass();
  calcWeightError();
  calcBiasError();
}

void Neuron::fixWeight(){
  for(int i=0;i<previousLayerSize;i++){
    weight[i] += -weightError[i] * HyperParams::learningRate;
  }
}

void Neuron::fixBias(){
  bias -=  biasError*HyperParams::learningRate;
}

float Neuron::getOutput(){
  return output;
}

void Neuron::addDeltaNext(float d, float weight){
  deltaNext += d*weight;
}

void Neuron::setInput(float input){
  netValue = input;
}

void Neuron::setDeltaNext(int error){
  deltaNext =error;
}

  /* Should be called before feeding a new input.
   * reset deltaNext and netvalue */
void Neuron::cleanForNextInput(){
  deltaNext = 0;
  netValue = -bias;
}

void Neuron::cleanForNextBatch(){
  for(int i=0;i<previousLayerSize;i++)
    weightError[i] = 0;
  biasError = 0;
}

  /*calculate net value, called from feedfoward()*/
void Neuron::calcNet(){
  // netValue = -bias;
  for(int i=0; i<previousLayerSize;i++){
    netValue += weight[i] * previousLayerNeurons[i]->getOutput();
  }
}

  /*calculate output, called from feedfoward*/
void Neuron::calcOutput(){
  output = HyperParams::activate(netValue);
}

  /* calculates delta*/
void Neuron::calcDelta(){
  delta = deltaNext * HyperParams::dactivate(netValue);
}

  /* pass delta to previous layer*/
void Neuron::backPass(){
  for(int i=0;i<previousLayerSize;i++){
    previousLayerNeurons[i]->addDeltaNext(delta,weight[i]);
  }
}


  /*calculates the error of weight*/
void Neuron::calcWeightError(){
  for(int i=0;i<previousLayerSize;i++){
    weightError[i] += delta * previousLayerNeurons[i]->getOutput();
  }
}
void Neuron::calcBiasError(){
  biasError -= delta;
}
