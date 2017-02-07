#include "imageloader.hpp"
#include <stdio.h>
#include <assert.h>

#define IMAGE_FILE_HEADER_SIZE 16
#define LABEL_FILE_HEADER_SIZE 8
#define IMAGE_FILE_NAME "trainimages"
#define LABEL_FILE_NAME "trainlabels"
#define IMAGE_WIDTH 28
#define IMAGE_HEIGHT 28

int imageloader::imageCount = 0;
unsigned char imageloader::pixels[IMAGE_HEIGHT][IMAGE_WIDTH] ={{0}};
unsigned char imageloader::label = 0;
FILE *imageloader::imageFile = NULL;
FILE *imageloader::labelFile = NULL;

double imageloader::getPixel(int row, int col){
	return ((double)pixels[row][col])/255;
}

void imageloader::openFile(){
	imageFile = fopen(IMAGE_FILE_NAME,"rb");
	assert(imageFile != NULL);
	labelFile =fopen(LABEL_FILE_NAME,"rb");
	assert(labelFile != NULL);

}

void imageloader::closeFile(){
	fclose(imageFile);
	fclose(labelFile);
}

/* loads the next image 
 * returns 0 if reaches EOF */
int imageloader::loadNextImage(){
	assert(imageFile != NULL);
	assert(labelFile != NULL);

	/* read pixels from file to buffer, then from buffer to imageloader::pixels */
	fseek(imageFile,IMAGE_FILE_HEADER_SIZE + IMAGE_WIDTH*IMAGE_HEIGHT*imageCount,SEEK_SET);
	unsigned char buffer[IMAGE_HEIGHT*IMAGE_WIDTH];
	int i =fread(buffer,sizeof(unsigned char),IMAGE_WIDTH*IMAGE_HEIGHT,imageFile);
	for(int r=0;r<IMAGE_HEIGHT;r++){
		for(int c=0;c<IMAGE_WIDTH;c++){
			pixels[r][c] = buffer[r*IMAGE_HEIGHT+c];
		}
	}

	/* read label from file */
	fseek(labelFile,LABEL_FILE_HEADER_SIZE + imageCount,SEEK_SET);
	int success = fread(&label,sizeof(unsigned char),1,labelFile);
  
	if(success == 0){
		return 0;
		assert(0);
	}
	else{
		imageCount++;
		return 1;
	}
}

unsigned char imageloader::getLabel(){
	return label;
}

void imageloader::rewind(int imagesBack){
	imageCount -= imagesBack;
}
