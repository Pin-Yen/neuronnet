class HyperParams
{
public:
	static double activate(double netValue);

  static double dactivate(double netValue);

  static double costFunction(double neuronOutput, double desiredOutput);
  static const double learningRate;
  static const int batchSize;
  static const int numberOfInputs;
  static const int hiddenLayerSize;
  static const int repeatTimes;
  static const int testSize;
};