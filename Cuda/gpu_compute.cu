#include <typeinfo>
#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <cuda.h>
#include <cuda_runtime.h>
#include <string>
#include <algorithm>
#include <vector>
#include <iterator>
using namespace std;

// kernel to cube
__global__ void cube(float *d_out, float *d_in){
  int idx = threadIdx.x;
  float f = d_in[idx];
  d_out[idx] = f*f*f;
}

void run_cube() {
  const int ARRAY_SIZE = 128;
  const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);
  float h_in[ARRAY_SIZE], h_out[ARRAY_SIZE];

  for(int i=0;i<ARRAY_SIZE;i++)
      h_in[i] = float(i);

  float *d_in, *d_out;
  cudaMalloc((void **) &d_in, ARRAY_BYTES);
  cudaMalloc((void **) &d_out, ARRAY_BYTES);

  cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice); // send data from CPU to GPU
  cube <<<1, ARRAY_SIZE>>> (d_out,d_in); // lauch kernel BLOCKS, THREADS
  cudaMemcpy(h_out, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost);  // send data from CPU to GPU

  // print the results
	for (int i =0; i < ARRAY_SIZE; i++) {
		printf("%f", h_out[i]);
		printf(((i % 4) != 3) ? "\t" : "\n");
	}
	cudaFree(d_in);
	cudaFree(d_out);
}

cv::Mat imageRGBA;
cv::Mat imageGrey;
uchar4        *d_rgbaImage__;
unsigned char *d_greyImage__;
size_t numRows() { return imageRGBA.rows; }
size_t numCols() { return imageRGBA.cols; }

void preProcess(uchar4 **inputImage, unsigned char **greyImage,
                uchar4 **d_rgbaImage, unsigned char **d_greyImage,
                string &filename) {
  cudaFree(0);
  cv::Mat image;
  image = cv::imread(filename.c_str(), CV_LOAD_IMAGE_COLOR);
  if (image.empty()) {
    cerr << "Couldn't open file: " << filename << endl;
    exit(1);
  }
  cv::cvtColor(image, imageRGBA, CV_BGR2RGBA);
  //allocate memory for the cv::Mat output
  imageGrey.create(image.rows, image.cols, CV_8UC1);
  if (!imageRGBA.isContinuous() || !imageGrey.isContinuous()) {
    cerr << "Images aren't continuous!! Exiting." << endl;
    exit(1);
  }

  *inputImage = (uchar4 *)imageRGBA.ptr<unsigned char>(0);
  *greyImage  = imageGrey.ptr<unsigned char>(0);

  const size_t numPixels = numRows() * numCols();
  //allocate memory on the device
  cudaMalloc(d_rgbaImage, sizeof(uchar4) * numPixels);
  cudaMalloc(d_greyImage, sizeof(unsigned char) * numPixels);
  cudaMemset(*d_greyImage, 0, numPixels * sizeof(unsigned char)); //make sure no memory is left laying around
  // copy input into GPU input
  cudaMemcpy(*d_rgbaImage, *inputImage, sizeof(uchar4) * numPixels, cudaMemcpyHostToDevice);
  // save to global variables
  d_rgbaImage__ = *d_rgbaImage;
  d_greyImage__ = *d_greyImage;
}

void postProcess(const string& output_file) {
  const int numPixels = numRows() * numCols();
  //copy the output back to the host
  cudaMemcpy(imageGrey.ptr<unsigned char>(0), d_greyImage__, sizeof(unsigned char) * numPixels, cudaMemcpyDeviceToHost);
  //output the image
  cv::imwrite(output_file.c_str(), imageGrey);
  //cleanup
  cudaFree(d_rgbaImage__);
  cudaFree(d_greyImage__);
}

__global__ void rgba_to_greyscale(const uchar4* const rgbaImage,
                unsigned char* const greyImage, int numRows, int numCols) {
  int index_x = blockIdx.x * blockDim.x + threadIdx.x;
  int index_y = blockIdx.y * blockDim.y + threadIdx.y;

  int grid_width = gridDim.x * blockDim.x;
  int index = index_y * grid_width + index_x;
  greyImage[index] =  .299f * rgbaImage[index].x + .587f * rgbaImage[index].y + .114f * rgbaImage[index].z;
}

void your_rgba_to_greyscale(const uchar4 * const h_rgbaImage, uchar4 * const d_rgbaImage,
                unsigned char* const d_greyImage, size_t numRows, size_t numCols) {
  const int thread = 16;
  const dim3 blockSize( thread, thread, 1);
  const dim3 gridSize( ceil(numRows/(float)thread), ceil(numCols/(float)thread), 1);
  rgba_to_greyscale<<<gridSize, blockSize>>>(d_rgbaImage, d_greyImage, numRows, numCols);
  cudaDeviceSynchronize();
}

int main(int argc, char* argv[]){
  // run_cube(); test function that calc cude of n numbers
  uchar4        *h_rgbaImage, *d_rgbaImage;
  unsigned char *h_greyImage, *d_greyImage;

  string input_file = argv[1];
  string output_file = "op_c.jpg";
  preProcess(&h_rgbaImage, &h_greyImage, &d_rgbaImage, &d_greyImage, input_file);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  your_rgba_to_greyscale(h_rgbaImage, d_rgbaImage, d_greyImage, numRows(), numCols());
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaDeviceSynchronize();
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  cout<<"Runtime (in ms) : "<<milliseconds<<endl;

  postProcess(output_file);
	return 0;
}
// Sample output:
// Runtime (in ms) : 0.033344

// nvcc `pkg-config --cflags --libs opencv` -o gpu_compute.o gpu_compute.cu
// ./gpu_compute.o cinque_terre_small.jpg
