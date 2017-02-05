class Network;

class Trainer
{
public:
	Trainer();
	~Trainer();
	
	/* trains the network */
	static void train(Network *network);

private:	
	static float outputErrorCache[10];
	static float batchTrainingError;

  /* updates the error cache AND training error*/ 
	static void updateErrorCache(Network *network);

	/* calculates the average error of each input CASE,
	 * and adds to batch error */
	static void calcAverageError();

	/* calculates the average error of each BATCH 
	 * called at the end of each batch learning process */
	static void calcAverageBatchError();

  /* logs the batchError to screen */
	static void logBatchError(int batchCount);

	/* cleans the outputErrorCache and batchError,
	 * called at the end of each batch cycle */
	static void cleanError();

};