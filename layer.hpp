class Neuron;

#define LAYERSIZE_MAX 28*28
class Layer
{

public:
  /* constructs a layer with size `size`,
   * allocates memory for neurons */
	Layer(int size);

  /* frees the memory pointed by neurons */
  ~Layer();

  /* intialize bias and weight of each neuron */
  void initialize();

  /* cleanup for next input*/
  void cleanForNextInput();

  /* cleanup for next batch*/
  void cleanForNextBatch();

  /* fetch input from imageloader*/
  void fetchInput();

  void feedFoward();

  void backProp();

  /* called in Network constructor,
   * gives the current layer a reference to the previous layer
   ** NOTICE : this only establishes a one-way relationship,
   ** should call connectNextLayer on the previous layer in order to est. a complete connection */
  void connectPreviousLayer(Layer *previousLayer);

  /* give the current layer a reference to the next layer,
   * refer to documentation of connectPreviousLayer for more details */  
  void connectNextLayer(Layer *nextlayer);

  /* set Error value of each output neuron */
  void setError(double errorArr[10]);

  /* fix weight and bias */
  void fix();

  /* returns a pointer of a neuron in this layer,
   * index: index in layer */
  Neuron* getNeuronPtr(int index);
  int getLayerSize();

  int getAnswer();

private:
  Neuron *neurons[LAYERSIZE_MAX];
  int layerSize;
};