class HyperParams
{
public:
	static float activate(float netValue);

  static float dactivate(float netValue);

  static float costFunction(float neuronOutput, float desiredOutput);
  static const float learningRate;
  static const int batchSize;
  static const int numberOfInputs;
  static const int hiddenLayerSize;
};