#define IMAGE_WIDTH 28
#define IMAGE_HEIGHT 28
#include <stdio.h>

class imageloader
{
public:
	imageloader();
	~imageloader();

	static unsigned char getPixel(int row, int col);
	static int loadNextImage();
	static unsigned char getLabel();
	static void openFile();
	static void closeFile();
	static int imageCount; 
private:
	static unsigned char pixels[IMAGE_HEIGHT][IMAGE_WIDTH];
	static unsigned char label;
	static FILE *imageFile, *labelFile;
};