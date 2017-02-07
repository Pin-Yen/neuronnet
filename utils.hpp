#include <time.h>
class Utils
{
public:
	Utils();
	~Utils();
	
  /* returns a random double between -0.5 and 0.5*/
  static double randdouble();
	static void printElapsedTime(time_t *start, time_t *end);
	static void writeLogFile(time_t *end, float accurarcy);
private:
	static bool randSeedInitialized;

  /* set the random seed, called by randdouble.*/
	static void randSetSeed();

};
