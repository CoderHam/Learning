#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <cuda.h>
#include <string>
#include <algorithm>
#include <vector>
#include <iterator>
using namespace std;

__global__ void gaussian_blur(const unsigned char* const inputChannel,
                   unsigned char* const outputChannel,
                   int numRows, int numCols,
                   const float* const filter, const int filterWidth) {
  int index_x = blockIdx.x * blockDim.x + threadIdx.x;
  int index_y = blockIdx.y * blockDim.y + threadIdx.y;

  if (index_x >= numCols || index_y >= numRows)
      return;
  float c = 0.0f;

  for (int fx = 0; fx < filterWidth; fx++) {
    for (int fy = 0; fy < filterWidth; fy++) {
      int imagex = index_x + fx - filterWidth / 2;
      int imagey = index_y + fy - filterWidth / 2;
      imagex = min(max(imagex,0),numCols-1);
      imagey = min(max(imagey,0),numRows-1);
      c += (filter[fy*filterWidth+fx] * inputChannel[imagey*numCols+imagex]);
    }
  }

  outputChannel[index_y*numCols + index_x] = c;
}

//This kernel takes in an image represented as a uchar4 and splits it into three images consisting of only one color channel each
__global__ void separateChannels(const uchar4* inputImageRGBA,
                      int numRows,int numCols,
                      unsigned char* redChannel, unsigned char* greenChannel, unsigned char* blueChannel) {
  int index_x = blockIdx.x * blockDim.x + threadIdx.x;
  int index_y = blockIdx.y * blockDim.y + threadIdx.y;

  if (index_x >= numCols || index_y >= numRows)
      return;

  int index = index_y * numCols + index_x;

  redChannel[index] = inputImageRGBA[index].x;
  greenChannel[index] = inputImageRGBA[index].y;
  blueChannel[index] = inputImageRGBA[index].z;
}

//This kernel takes in three color channels and recombines them into one image.  The alpha channel is set to 255 for no transparency.
__global__ void recombineChannels(const unsigned char* redChannel, const unsigned char* greenChannel, const unsigned char* blueChannel,
                       uchar4* outputImageRGBA,
                       int numRows, int numCols) {
  const int2 thread_2D_pos = make_int2( blockIdx.x * blockDim.x + threadIdx.x,
                                        blockIdx.y * blockDim.y + threadIdx.y);

  const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

  if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows)
    return;

  unsigned char red   = redChannel[thread_1D_pos];
  unsigned char green = greenChannel[thread_1D_pos];
  unsigned char blue  = blueChannel[thread_1D_pos];

  //Alpha should be 255 for no transparency
  uchar4 outputPixel = make_uchar4(red, green, blue, 255);
  outputImageRGBA[thread_1D_pos] = outputPixel;
}

unsigned char *d_red, *d_green, *d_blue;
float         *d_filter;

void allocateMemoryAndCopyToGPU(const size_t numRowsImage, const size_t numColsImage,
                                const float* const h_filter, const size_t filterWidth) {
  //allocate memory for the three different channels
  cudaMalloc(&d_red,   sizeof(unsigned char) * numRowsImage * numColsImage);
  cudaMalloc(&d_green, sizeof(unsigned char) * numRowsImage * numColsImage);
  cudaMalloc(&d_blue,  sizeof(unsigned char) * numRowsImage * numColsImage);

  //Allocate memory for the filter on the GPU
  cudaMalloc(&d_filter, sizeof(float) * filterWidth * filterWidth);

  //Copy the filter on the host (h_filter) to the memory you just allocated on the GPU
  cudaMemcpy(d_filter, h_filter, sizeof(float) * filterWidth * filterWidth, cudaMemcpyHostToDevice);
}

void your_gaussian_blur(const uchar4 * const h_inputImageRGBA, uchar4 * const d_inputImageRGBA,
                        uchar4* const d_outputImageRGBA, const size_t numRows, const size_t numCols,
                        unsigned char *d_redBlurred, unsigned char *d_greenBlurred, unsigned char *d_blueBlurred,
                        const int filterWidth) {
  // Set block size (i.e., number of threads per block)
  const dim3 blockSize(16,16,1);

  //Compute grid size (i.e., number of blocks per kernel launch) from the image size and and block size.
  const dim3 gridSize(numCols/blockSize.x+1,numRows/blockSize.y+1,1);

  // Launch kernel for separating the RGBA image into different color channels
  separateChannels<<<gridSize, blockSize>>>(d_inputImageRGBA,
                                            numRows,numCols,
                                            d_red,d_green,d_blue);
  cudaDeviceSynchronize(); cudaGetLastError();

  // Call the convolution/blur kernel here 3 times, once for each color channel.
  gaussian_blur<<<gridSize, blockSize>>>(d_red,d_redBlurred,numRows,numCols,d_filter,filterWidth);
  gaussian_blur<<<gridSize, blockSize>>>(d_green,d_greenBlurred,numRows,numCols,d_filter,filterWidth);
  gaussian_blur<<<gridSize, blockSize>>>(d_blue,d_blueBlurred,numRows,numCols,d_filter,filterWidth);

  cudaDeviceSynchronize(); cudaGetLastError();

  // Recombine results
  recombineChannels<<<gridSize, blockSize>>>(d_redBlurred, d_greenBlurred, d_blueBlurred,
                                             d_outputImageRGBA,
                                             numRows, numCols);
  cudaDeviceSynchronize(); cudaGetLastError();
}

//Free all the memory that we allocated
void cleanup() {
  cudaFree(d_red);
  cudaFree(d_green);
  cudaFree(d_blue);
}

void channelConvolution(const unsigned char* const channel,
                        unsigned char* const channelBlurred,
                        const size_t numRows, const size_t numCols,
                        const float *filter, const int filterWidth)
{
  //Dealing with an even width filter is trickier
  assert(filterWidth % 2 == 1);

  //For every pixel
  for (int r = 0; r < (int)numRows; ++r) {
    for (int c = 0; c < (int)numCols; ++c) {
      float result = 0.f;
      //For every value in the filter around the pixel (c, r)
      for (int filter_r = -filterWidth/2; filter_r <= filterWidth/2; ++filter_r) {
        for (int filter_c = -filterWidth/2; filter_c <= filterWidth/2; ++filter_c) {
          //Find the global image position for this filter position
          //clamp to boundary of the image
          int image_r = min(max(r + filter_r, 0), static_cast<int>(numRows - 1));
          int image_c = min(max(c + filter_c, 0), static_cast<int>(numCols - 1));

          float image_value = static_cast<float>(channel[image_r * numCols + image_c]);
          float filter_value = filter[(filter_r + filterWidth/2) * filterWidth + filter_c + filterWidth/2];

          result += image_value * filter_value;
        }
      }

      channelBlurred[r * numCols + c] = result;
    }
  }
}

void referenceCalculation(const uchar4* const rgbaImage, uchar4 *const outputImage,
                          size_t numRows, size_t numCols,
                          const float* const filter, const int filterWidth)
{
  unsigned char *red   = new unsigned char[numRows * numCols];
  unsigned char *blue  = new unsigned char[numRows * numCols];
  unsigned char *green = new unsigned char[numRows * numCols];

  unsigned char *redBlurred   = new unsigned char[numRows * numCols];
  unsigned char *blueBlurred  = new unsigned char[numRows * numCols];
  unsigned char *greenBlurred = new unsigned char[numRows * numCols];

  //First we separate the incoming RGBA image into three separate channels
  //for Red, Green and Blue
  for (size_t i = 0; i < numRows * numCols; ++i) {
    uchar4 rgba = rgbaImage[i];
    red[i]   = rgba.x;
    green[i] = rgba.y;
    blue[i]  = rgba.z;
  }

  //Now we can do the convolution for each of the color channels
  channelConvolution(red, redBlurred, numRows, numCols, filter, filterWidth);
  channelConvolution(green, greenBlurred, numRows, numCols, filter, filterWidth);
  channelConvolution(blue, blueBlurred, numRows, numCols, filter, filterWidth);

  //now recombine into the output image - Alpha is 255 for no transparency
  for (size_t i = 0; i < numRows * numCols; ++i) {
    uchar4 rgba = make_uchar4(redBlurred[i], greenBlurred[i], blueBlurred[i], 255);
    outputImage[i] = rgba;
  }

  delete[] red;
  delete[] green;
  delete[] blue;

  delete[] redBlurred;
  delete[] greenBlurred;
  delete[] blueBlurred;
}

cv::Mat imageInputRGBA;
cv::Mat imageOutputRGBA;

uchar4 *d_inputImageRGBA__;
uchar4 *d_outputImageRGBA__;

float *h_filter__;

size_t numRows() { return imageInputRGBA.rows; }
size_t numCols() { return imageInputRGBA.cols; }

//return types are void since any internal error will be handled by quitting
//no point in returning error codes...
//returns a pointer to an RGBA version of the input image
//and a pointer to the single channel grey-scale output
//on both the host and device
void preProcess(uchar4 **h_inputImageRGBA, uchar4 **h_outputImageRGBA,
                uchar4 **d_inputImageRGBA, uchar4 **d_outputImageRGBA,
                unsigned char **d_redBlurred,
                unsigned char **d_greenBlurred,
                unsigned char **d_blueBlurred,
                float **h_filter, int *filterWidth,
                const string &filename) {

  //make sure the context initializes ok
  cudaFree(0);

  cv::Mat image = cv::imread(filename.c_str(), CV_LOAD_IMAGE_COLOR);
  if (image.empty()) {
    cerr << "Couldn't open file: " << filename << endl;
    exit(1);
  }

  cv::cvtColor(image, imageInputRGBA, CV_BGR2RGBA);

  //allocate memory for the output
  imageOutputRGBA.create(image.rows, image.cols, CV_8UC4);

  //This shouldn't ever happen given the way the images are created
  //at least based upon my limited understanding of OpenCV, but better to check
  if (!imageInputRGBA.isContinuous() || !imageOutputRGBA.isContinuous()) {
    cerr << "Images aren't continuous!! Exiting." << endl;
    exit(1);
  }

  *h_inputImageRGBA  = (uchar4 *)imageInputRGBA.ptr<unsigned char>(0);
  *h_outputImageRGBA = (uchar4 *)imageOutputRGBA.ptr<unsigned char>(0);

  const size_t numPixels = numRows() * numCols();
  //allocate memory on the device for both input and output
  cudaMalloc(d_inputImageRGBA, sizeof(uchar4) * numPixels);
  cudaMalloc(d_outputImageRGBA, sizeof(uchar4) * numPixels);
  cudaMemset(*d_outputImageRGBA, 0, numPixels * sizeof(uchar4)); //make sure no memory is left laying around

  //copy input array to the GPU
  cudaMemcpy(*d_inputImageRGBA, *h_inputImageRGBA, sizeof(uchar4) * numPixels, cudaMemcpyHostToDevice);

  d_inputImageRGBA__  = *d_inputImageRGBA;
  d_outputImageRGBA__ = *d_outputImageRGBA;

  //now create the filter that they will use
  const int blurKernelWidth = 9; // 9x9 filter
  const float blurKernelSigma = 2.;

  *filterWidth = blurKernelWidth;

  //create and fill the filter we will convolve with
  *h_filter = new float[blurKernelWidth * blurKernelWidth];
  h_filter__ = *h_filter;

  float filterSum = 0.f; //for normalization

  for (int r = -blurKernelWidth/2; r <= blurKernelWidth/2; ++r) {
    for (int c = -blurKernelWidth/2; c <= blurKernelWidth/2; ++c) {
      float filterValue = expf( -(float)(c * c + r * r) / (2.f * blurKernelSigma * blurKernelSigma));
      (*h_filter)[(r + blurKernelWidth/2) * blurKernelWidth + c + blurKernelWidth/2] = filterValue;
      filterSum += filterValue;
    }
  }

  float normalizationFactor = 1.f / filterSum;

  for (int r = -blurKernelWidth/2; r <= blurKernelWidth/2; ++r) {
    for (int c = -blurKernelWidth/2; c <= blurKernelWidth/2; ++c) {
      (*h_filter)[(r + blurKernelWidth/2) * blurKernelWidth + c + blurKernelWidth/2] *= normalizationFactor;
    }
  }

  cudaMalloc(d_redBlurred,    sizeof(unsigned char) * numPixels);
  cudaMalloc(d_greenBlurred,  sizeof(unsigned char) * numPixels);
  cudaMalloc(d_blueBlurred,   sizeof(unsigned char) * numPixels);
  cudaMemset(*d_redBlurred,   0, sizeof(unsigned char) * numPixels);
  cudaMemset(*d_greenBlurred, 0, sizeof(unsigned char) * numPixels);
  cudaMemset(*d_blueBlurred,  0, sizeof(unsigned char) * numPixels);
}

void postProcess(const string& output_file) {
  const int numPixels = numRows() * numCols();
  //copy the output back to the host
  cudaMemcpy(imageOutputRGBA.ptr<unsigned char>(0), d_outputImageRGBA__, sizeof(uchar4) * numPixels, cudaMemcpyDeviceToHost);

  cv::Mat imageOutputBGR;
  cv::cvtColor(imageOutputRGBA, imageOutputBGR, CV_RGBA2BGR);
  //output the image
  cv::imwrite(output_file.c_str(), imageOutputBGR);

  //cleanup
  cudaFree(d_inputImageRGBA__);
  cudaFree(d_outputImageRGBA__);
  delete[] h_filter__;
}

int main(int argc, char **argv) {
  uchar4 *h_inputImageRGBA,  *d_inputImageRGBA;
  uchar4 *h_outputImageRGBA, *d_outputImageRGBA;
  unsigned char *d_redBlurred, *d_greenBlurred, *d_blueBlurred;

  float *h_filter;
  int    filterWidth;

  string input_file;
  string output_file;
  if (argc == 3) {
    input_file  = string(argv[1]);
    output_file = string(argv[2]);
  }
  else {
    cerr << "Usage: ./hw input_file output_file" << endl;
    exit(1);
  }
  //load the image and give us our input and output pointers
  preProcess(&h_inputImageRGBA, &h_outputImageRGBA, &d_inputImageRGBA, &d_outputImageRGBA,
             &d_redBlurred, &d_greenBlurred, &d_blueBlurred,
             &h_filter, &filterWidth, input_file);

  allocateMemoryAndCopyToGPU(numRows(), numCols(), h_filter, filterWidth);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  //call the students' code
  your_gaussian_blur(h_inputImageRGBA, d_inputImageRGBA, d_outputImageRGBA, numRows(), numCols(),
                     d_redBlurred, d_greenBlurred, d_blueBlurred, filterWidth);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaDeviceSynchronize();
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  cout<<"Runtime (in ms) : "<<milliseconds<<endl;

  cleanup();
  //check results and output the blurred image
  postProcess(output_file);

  cudaFree(d_redBlurred);
  cudaFree(d_greenBlurred);
  cudaFree(d_blueBlurred);

  return 0;
}

// Sample output:
// Runtime (in ms) : 1.5575

// nvcc `pkg-config --cflags --libs opencv` -o img_blur_gpu.o img_blur_gpu.cu
// ./img_blur_gpu.o cinque_terre_small.jpg blur.jpg
