#define LAYERSIZE_MAX 28*28
class Layer;

class Neuron{
public:
  Neuron();

  /* initialize weight and bias to a random value */
  void initialize();

  /* called by Layer::connectPreviousLayer
   * refer to Layer::connectPreviousLayer for details */
  void connectPreviousLayer(Layer *previousLayer);

  /* called by Layer::connectNextLayer
   * refer to Layer::connectNextLayer for details */
  void connectNextLayer(Layer *nextLayer);

  /* ONLY BE CALLED WHEN RESETTING THE WHOLE TRAINING PROCESS
   * sets All variables to 0, except those determined by the network structure (i.e. nextLayer, previousLayer)
   * not to be confused with clearForNextInput */
  void resetAll();

  void feedFoward();

  void backProp();

  void fix();

  double getOutput();

  /* called by neurons of the next layer, in backpass()
   * passively collecting the delta of the next layer. */
  void addDeltaNext(double d, double weight);

  /* called by Layer::fetchInput(), 
   * sets the input value of the input neurons */
  void setInput(double input);

  /* indirectly called by Network::setError()
   * set error of output neurons,*/ 
  void setDeltaNext(double error);

  /* Should be called before feeding a new input.
   * reset deltaNext and netValue */
  void cleanForNextInput();

  /* Should be called before each new batch,
   * resets weigthError and Bias Error*/
  void cleanForNextBatch();

private:
  /*calculate net value, called from feedfoward()*/
  void calcNet();

  /*calculate output, called from feedfoward*/
  void calcOutput();

  /* calculates delta*/
  void calcDelta();

  /* pass delta to previous layer*/
  void backPass();

  /*calculates the error of weight*/
  void calcWeightError();

  /* calculates bias Error*/
  void calcBiasError();

  /* the following members should be initialized */  
  int previousLayerSize, nextLayerSize;
  /* the following members should be initialized */  


  double netValue , output;
  double bias;
  double delta, deltaNext;
  double weight[LAYERSIZE_MAX];
  double weightError[LAYERSIZE_MAX];
  double biasError;
  Neuron *previousLayerNeurons[LAYERSIZE_MAX];
  Neuron *nextLayerNeurons[LAYERSIZE_MAX];
};
