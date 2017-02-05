class Utils
{
public:
	Utils();
	~Utils();
	
  /* returns a random float between -0.5 and 0.5*/
  static float randFloat();

private:
	static bool randSeedInitialized;

  /* set the random seed, called by randFloat.*/
	static void randSetSeed();

};
