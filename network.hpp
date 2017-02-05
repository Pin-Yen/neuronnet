#include "layer.hpp"

class Network
{
public:
	/* constructor for 3 layer network,
	 * allocate layer pointer,
	 * establish connection between layers */
	Network(int inputLayerSize, int hiddenLayerSize, int outputLayerSize);

	/* deallocate layer */
	~Network();

	/* intialize each layer,
	 * i.e. set weights and bias of neurons of every layer to a random value*/
	void initialize();

	/* called before every new input
	 * cleans of the mess done by the previous input*/
	void cleanForNextInput();

	/* makes input layer fetch input from imageloader */
	void fetchInput();

	void feedFoward();

	void backProp();

	/* fix weight of each synapse by gradient descent*/
	void fixWeight();

	void fixBias();
	
	/* returns the output value of an output neuron
	 * index: index of neuron in the output layer
	 * called at multiple place by the Network class*/
	float getOutput(int index);

	/* set ErrorValue to the output neuron, 
	 * called once a batch */
	void setError(float errorArr[10]);

private:
	Layer *inputLayer, *hiddenLayer, *outputLayer;
};