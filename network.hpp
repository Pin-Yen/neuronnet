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

	/* called before each new batch */
	void cleanForNextBatch();

	void feedFoward();

	void backProp();

	/* fix weight and bias of each synapse by gradient descent*/
	void fix();
	
	/* returns the output value of an output neuron
	 * index: index of neuron in the output layer
	 * called at multiple place by the Network class*/
	double getOutput(int index);

	/* set ErrorValue to the output neuron, 
	 * called once a batch */
	void setError(double errorArr[10]);

	/* get NN prediction */
	int getAnswer();

private:
	Layer *inputLayer, *hiddenLayer, *outputLayer;
};