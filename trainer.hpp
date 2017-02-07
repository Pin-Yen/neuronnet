class Network;

class Trainer
{
public:
	Trainer();
	~Trainer();
	
	/* trains the network */
	static void train(Network *network);
	static float test(Network *network);

private:	
	static double outputErrorCache[10];
	static double batchTrainingError;

	/* records the number of cases successfully classified */
	static int rightCount;

  /* updates the error cache AND training error*/ 
	static void updateErrorCache(Network *network);

	/* calculates the average error of each input CASE,
	 * and adds to batch error */
	// static void calcAverageError();

	/* calculates batch error and
   * logs the batchError to screen */
	static void logBatchError(int batchCount);

	/* cleans the outputErrorCache and batchError,
	 * called at the end of each batch cycle */
	static void cleanBatchError();

	static void logBasicInfo();

	static void printPredictDetail(Network *network);


};