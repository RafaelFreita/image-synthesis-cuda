
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iterator>
#include <limits>
#include <numeric>
#include <string>

#include <stb_image.h>
#include "stb_image_write.h"

#include "Timer.hpp"
#include "cxxopts.hpp"

struct Image {
	int width, height, channelsQtd;
	uint8_t* data;

	bool loadedImage = false;
	bool cleaned = false;

	bool Load(char const* path) {
		data = stbi_load(path, &width, &height, &channelsQtd, 0);
		if (!data) {
			FreeImage();
			return false;
		}

		loadedImage = true;
		return true;
	}

	void FreeImage() {
		if (cleaned || !loadedImage) return;
		cleaned = true;
		stbi_image_free(data);
	}

	int GetDataSize() {
		return width * height * channelsQtd;
	}

	void AllocateDataArray() {
		data = new uint8_t[width * height * channelsQtd];
	}

	void WriteImage(char const* path) {
		stbi_write_png(path, width, height, channelsQtd, data, width * channelsQtd);
	}

	~Image() {
		if (!cleaned) FreeImage();
	}

};

#pragma region Helpers

float distanceBetweenColors(uint8_t r1, uint8_t g1, uint8_t b1, uint8_t r2, uint8_t g2, uint8_t b2) {
	int16_t r = r2 - r1;
	int16_t g = g2 - g1;
	int16_t b = b2 - b1;
	float d = r * r + g * g + b * b;
	return sqrt(d);
}

__device__ float distanceBetweenColorsCuda(uint8_t r1, uint8_t g1, uint8_t b1, uint8_t r2, uint8_t g2, uint8_t b2) {
	int16_t r = r2 - r1;
	int16_t g = g2 - g1;
	int16_t b = b2 - b1;
	float d = r * r + g * g + b * b;
	return sqrt(d);
}

uint16_t wrapValue(uint16_t a, uint16_t b) {
	if (a < 0) return b + a;
	else if (a >= b) return abs(b - a);
	return a;
}

double computeAverage(std::vector<double>& v)
{
	size_t n = v.size();
	if (n == 0)
		return 0.0;

	return std::accumulate(v.begin(), v.end(), 0.0) / n;
}

double computeMedian(std::vector<double>& v)
{
	size_t n = v.size() / 2;
	std::nth_element(v.begin(), v.begin() + n, v.end()); // Like std::sort, but only sorts until nth element
	return v[n];
}

#pragma endregion

#pragma region CPU

unsigned char* synthetiseImage(Image& sampleImage, Image& preSynImage, uint8_t neighborhoodSize) {

	if (neighborhoodSize % 2 == 0) {
		printf("ERROR neighborhoodSize for image synthesis must be odd!\n");
		return nullptr;
	}

	int arrSize = preSynImage.width * preSynImage.height * 3;
	unsigned char* synthetizedImage = new unsigned char[arrSize];

	for (int i = 0; i < arrSize; i++)
	{
		synthetizedImage[i] = preSynImage.data[i];
	}

	int neighborSynX = 0, neighborSynY = 0;
	int neighborSampleX = 0, neighborSampleY = 0;

	uint16_t winnerPixelPosX, winnerPixelPosY;
	float winnerPontuation;
	float pontuation = 0;

	uint8_t neighborhoodHalf = (neighborhoodSize / 2);

	// For every pixel
	for (int y = 0; y < preSynImage.height; y++)
	{
		for (int x = 0; x < preSynImage.width; x++)
		{

			winnerPontuation = std::numeric_limits<float>::max();

			// For every pixel in sample image
			for (int sY = 0; sY < sampleImage.height; sY++)
			{
				for (int sX = 0; sX < sampleImage.width; sX++)
				{

					pontuation = 0.0f;

					// Check neighbors - From upper corner to the pixel itself
					for (int nY = -neighborhoodHalf; nY <= 0; nY++)
					{

						neighborSynY = y + nY;
						if (neighborSynY < 0) {
							neighborSynY = preSynImage.height + neighborSynY;
						}
						else if (neighborSynY >= preSynImage.height) {
							neighborSynY = abs(preSynImage.height - neighborSynY);
						}

						neighborSampleY = sY + nY;
						if (neighborSampleY < 0) {
							neighborSampleY = sampleImage.height + neighborSampleY;
						}
						else if (neighborSampleY >= sampleImage.height) {
							neighborSampleY = abs(sampleImage.height - neighborSampleY);
						}

						for (int nX = -neighborhoodHalf; nX <= neighborhoodHalf; nX++)
						{
							if (nY == 0 && nX > 0) { // Last line going only until the proper pixel
								break;
							}

							neighborSynX = x + nX;
							if (neighborSynX < 0) {
								neighborSynX = preSynImage.width + neighborSynX;
							}
							else if (neighborSynX >= preSynImage.width) {
								neighborSynX = abs(preSynImage.width - neighborSynX);
							}

							neighborSampleX = sX + nX;
							if (neighborSampleX < 0) {
								neighborSampleX = sampleImage.width + neighborSampleX;
							}
							else if (neighborSampleX >= sampleImage.width) {
								neighborSampleX = abs(sampleImage.width - neighborSampleX);
							}

							// Compare pixels from sample with preSyn
							// 0 - R // 1 - G // 2 - B

							int newImagePos = neighborSynX * preSynImage.channelsQtd + (neighborSynY * preSynImage.width * preSynImage.channelsQtd);
							int tempPos = neighborSampleX * sampleImage.channelsQtd + (neighborSampleY * sampleImage.width * sampleImage.channelsQtd);

							float dist = distanceBetweenColors(
								synthetizedImage[newImagePos + 0],
								synthetizedImage[newImagePos + 1],
								synthetizedImage[newImagePos + 2],

								sampleImage.data[tempPos + 0],
								sampleImage.data[tempPos + 1],
								sampleImage.data[tempPos + 2]
							);

							pontuation += dist;
						}
					}
					// End of neighbors

					// Assign new winner if pontution was smaller
					if (pontuation < winnerPontuation) {
						winnerPixelPosX = sX;
						winnerPixelPosY = sY;
						winnerPontuation = pontuation;
					}

				}
			}
			// End of sample check

			int pixelPos = x * preSynImage.channelsQtd + (y * preSynImage.width * preSynImage.channelsQtd);
			int winerPixelPos = winnerPixelPosX * sampleImage.channelsQtd + (winnerPixelPosY * sampleImage.width * sampleImage.channelsQtd);
			synthetizedImage[pixelPos + 0] = sampleImage.data[winerPixelPos + 0];
			synthetizedImage[pixelPos + 1] = sampleImage.data[winerPixelPos + 1];
			synthetizedImage[pixelPos + 2] = sampleImage.data[winerPixelPos + 2];

		}
	}

	return synthetizedImage;
}

#pragma endregion

#pragma region GPU
cudaError_t synthetiseImageCuda(
	uint16_t sampleWidth, uint16_t sampleHeight, uint16_t sampleChannels, uint8_t* sampleData,
	uint16_t preSynWidth, uint16_t preSynHeight, uint16_t preSynChannels, uint8_t* preSynData,
	uint8_t neighborhood, uint8_t* synthetizedData, uint8_t threadsPerBlock = 16U);

__global__ void synthetiseImageKernel(
	int16_t sampleWidth, int16_t sampleHeight, int16_t sampleChannels, uint8_t* sampleData,
	int16_t preSynWidth, int16_t preSynHeight, int16_t preSynChannels, uint8_t* synthetizedData,
	int16_t neighborhoodHalf, float* pontuationArray, int16_t x, int16_t y) {

	/*int arrSize = preSynWidth * preSynHeight * 3;
	unsigned char* newImage = new unsigned char[arrSize];

	for (int i = 0; i < arrSize; i++)
	{
		newImage[i] = preSynData[i];
	}*/

	int32_t neighborSynX = 0, neighborSynY = 0;
	int32_t neighborSampleX = 0, neighborSampleY = 0;

	float pontuation = 0;

	int16_t sX = threadIdx.x + blockIdx.x * blockDim.x;
	int16_t sY = threadIdx.y + blockIdx.y * blockDim.y;

	if (sX >= sampleWidth || sY >= sampleHeight) return;

	for (int nY = -neighborhoodHalf; nY <= 0; nY++)
	{

		neighborSynY = y + nY;
		if (neighborSynY < 0) neighborSynY = preSynHeight + neighborSynY;
		else if (neighborSynY >= preSynHeight) neighborSynY = abs(preSynHeight - neighborSynY);

		neighborSampleY = sY + nY;
		if (neighborSampleY < 0) neighborSampleY = sampleHeight + neighborSampleY;
		else if (neighborSampleY >= sampleHeight) neighborSampleY = abs(sampleHeight - neighborSampleY);

		for (int nX = -neighborhoodHalf; nX <= neighborhoodHalf; nX++)
		{
			if (nY == 0 && nX > 0) { // Last line going only until the proper pixel
				break;
			}

			neighborSynX = x + nX;
			if (neighborSynX < 0) neighborSynX = preSynWidth + neighborSynX;
			else if (neighborSynX >= preSynWidth) neighborSynX = abs(preSynWidth - neighborSynX);

			neighborSampleX = sX + nX;
			if (neighborSampleX < 0) neighborSampleX = sampleWidth + neighborSampleX;
			else if (neighborSampleX >= sampleWidth) neighborSampleX = abs(sampleWidth - neighborSampleX);

			// Compare pixels from sample with preSyn
			// 0 - R // 1 - G // 2 - B

			int newImagePos = neighborSynX * preSynChannels + (neighborSynY * preSynWidth * preSynChannels);
			int tempPos = neighborSampleX * sampleChannels + (neighborSampleY * sampleWidth * sampleChannels);

			float dist = distanceBetweenColorsCuda(
				synthetizedData[newImagePos + 0],
				synthetizedData[newImagePos + 1],
				synthetizedData[newImagePos + 2],

				sampleData[tempPos + 0],
				sampleData[tempPos + 1],
				sampleData[tempPos + 2]
			);

			pontuation += dist;
		}
	}

	int pontuationIndex = sX + sY * sampleWidth;
	pontuationArray[pontuationIndex] = pontuation;

}

#pragma endregion

int main(int argc, char* argv[])
{

#pragma region Command Line Options
	cxxopts::Options options("Cuda Image Synthesis", "Synthetyse an image using a sample and a pre-synthesis image with Cuda. \nRafael de Freitas, 2020\n");

	options.add_options()
		("h,help", "Print usage")
		("c,cpu", "Run on CPU", cxxopts::value<bool>()->default_value("false"))
		("g,gpu", "Run on GPU", cxxopts::value<bool>()->default_value("true"))
		("t,threads", "How many threads per block", cxxopts::value<int>()->default_value("16"))
		("n,neighborhood", "Neighborhood size - Must be odd", cxxopts::value<int>()->default_value("5"))
		("s,sample", "Sample image path", cxxopts::value<std::string>()->default_value(""))
		("p,presyn", "Pre-synthesis image path", cxxopts::value<std::string>()->default_value(""))
		("r,result", "Result image path WITHOUT EXTENSION", cxxopts::value<std::string>()->default_value(""))
		("i,itt", "How many tests to run", cxxopts::value<int>()->default_value("1"))
		;

	auto optionsResult = options.parse(argc, argv);

	if (optionsResult.count("help"))
	{
		std::cout << options.help() << std::endl;
		return 0;
	}

	const bool runCPU = optionsResult["cpu"].as<bool>();
	const bool runGPU = optionsResult["gpu"].as<bool>();

	const std::string pathSample = optionsResult["sample"].as<std::string>();
	const std::string pathPresyn = optionsResult["presyn"].as<std::string>();
	const std::string pathResult = optionsResult["result"].as<std::string>();

	const int neighborhoodSize = optionsResult["neighborhood"].as<int>();
	const int threadsPerBlock = optionsResult["threads"].as<int>();

	const int testIterations = optionsResult["itt"].as<int>();

	//
	// Checking requirements
	//
	if (pathSample == "") {
		printf("ERROR Can't run without sample path! \n");
		return 1;
	}

	if (pathPresyn == "") {
		printf("ERROR Can't run without pre-synthesis path! \n");
		return 1;
	}

	if (pathResult == "") {
		printf("ERROR Can't run without result path! \n");
		return 1;
	}

	if (neighborhoodSize % 2 == 0) {
		printf("ERROR NeighborhoodSize must be odd!\n");
		return 1;
	}

#pragma endregion

	//
	// Loading images
	//
	stbi_set_flip_vertically_on_load(true);

	Image sampleImg;
	sampleImg.Load(pathSample.c_str());

	Image preSynImg;
	preSynImg.Load(pathPresyn.c_str());

	Timer timer = Timer();
	std::vector<double> execTimesCPU, execTimesGPU;

	if (runCPU) {
		unsigned char* synData;

		for (int i = 0; i < testIterations; i++) {
			timer.start();
			synData = synthetiseImage(sampleImg, preSynImg, neighborhoodSize);
			if (synData == nullptr) {
				system("pause");
				return 1;
			}
			timer.finish();
			execTimesCPU.push_back(timer.getElapsedTimeMs());
		}

		Image synthetizedImageCPU;
		synthetizedImageCPU.width = preSynImg.width;
		synthetizedImageCPU.height = preSynImg.height;
		synthetizedImageCPU.channelsQtd = 3;
		synthetizedImageCPU.data = synData;

		if(!runGPU)
			synthetizedImageCPU.WriteImage((pathResult + ".png").c_str());
		else
			synthetizedImageCPU.WriteImage((pathResult + "-CPU.png").c_str());

		printf("CPU Average Time (ms): %f\n", computeAverage(execTimesCPU));
		printf("CPU Median Time (ms): %f\n", computeMedian(execTimesCPU));
		printf("\n");
	}

	if (runGPU) {
		Image synthetizedImageGPU;
		synthetizedImageGPU.width = preSynImg.width;
		synthetizedImageGPU.height = preSynImg.height;
		synthetizedImageGPU.channelsQtd = 3;
		synthetizedImageGPU.AllocateDataArray();

#pragma region Cuda calls
		for (int i = 0; i < testIterations; i++) {
			timer.start();
			cudaError_t cudaStatus = synthetiseImageCuda(
				sampleImg.width, sampleImg.height, sampleImg.channelsQtd, sampleImg.data,
				preSynImg.width, preSynImg.height, preSynImg.channelsQtd, preSynImg.data,
				5, synthetizedImageGPU.data, threadsPerBlock
			);
			timer.finish();

			execTimesGPU.push_back(timer.getElapsedTimeMs());

			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "synthetizeCuda failed!\n");
				system("pause");
				return 1;
			}

			cudaStatus = cudaDeviceReset();
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaDeviceReset failed!\n");
				system("pause");
				return 1;
			}
		}

		if (!runCPU)
			synthetizedImageGPU.WriteImage((pathResult + ".png").c_str());
		else
			synthetizedImageGPU.WriteImage((pathResult + "-GPU.png").c_str());

		printf("GPU Average Time (ms): %f\n", computeAverage(execTimesGPU));
		printf("GPU Median Time (ms): %f\n", computeMedian(execTimesGPU));
		printf("\n");
#pragma endregion

	}

	//
	// Cleanup
	//
	sampleImg.FreeImage();
	preSynImg.FreeImage();

	return 0;
}

cudaError_t synthetiseImageCuda(uint16_t sampleWidth, uint16_t sampleHeight, uint16_t sampleChannels, uint8_t* sampleData,
	uint16_t preSynWidth, uint16_t preSynHeight, uint16_t preSynChannels, uint8_t* preSynData,
	uint8_t neighborhood, uint8_t* synthetizedData, uint8_t threadsPerBlock) {

	const uint32_t preSynSize = preSynWidth * preSynHeight * preSynChannels;
	const size_t preSynSizeBytes = preSynSize * sizeof(uint8_t);
	const size_t sampleSizeBytes = sampleWidth * sampleHeight * sampleChannels * sizeof(uint8_t);

	uint8_t* cuda_sampleData = 0;
	uint8_t* cuda_synthetizedData = 0;

	const uint32_t pontuationsArraySize = sampleWidth * sampleHeight;
	float* pontuationsArray = new float[pontuationsArraySize];
	float* cuda_pontuationsArray = 0;

	for (int i = 0; i < preSynSize; i++)
	{
		synthetizedData[i] = preSynData[i];
	}

	uint8_t neighborhoodHalf = neighborhood / 2;
	cudaError_t cudaStatus;

#pragma region Cuda Memory Allocation

	// Set Cuda device
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Alocate arrays in GPU
	cudaStatus = cudaMalloc((void**)& cuda_synthetizedData, preSynSizeBytes);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)& cuda_sampleData, sampleSizeBytes);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)& cuda_pontuationsArray, sizeof(float) * pontuationsArraySize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy sample data
	cudaStatus = cudaMemcpy(cuda_sampleData, sampleData, sampleSizeBytes, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Copy synthesis data
	cudaStatus = cudaMemcpy(cuda_synthetizedData, synthetizedData, preSynSizeBytes, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

#pragma endregion

	dim3 threadsPerBlockDim(threadsPerBlock, threadsPerBlock);
	// Adding one block on each dimension for non-square images. The boundary check is inside the kernel.
	dim3 numBlocks(sampleWidth / threadsPerBlockDim.x + 1, sampleHeight / threadsPerBlockDim.y + 1);

	for (int y = 0; y < preSynHeight; y++)
	{
		for (int x = 0; x < preSynWidth; x++)
		{

			synthetiseImageKernel << < numBlocks, threadsPerBlockDim >> > (
				sampleWidth, sampleHeight, sampleChannels, cuda_sampleData,
				preSynWidth, preSynHeight, preSynChannels, cuda_synthetizedData,
				neighborhoodHalf, cuda_pontuationsArray, x, y);

			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "synthetiseImageKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
				goto Error;
			}

			cudaStatus = cudaDeviceSynchronize();
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaDeviceSynchronize returned error code %d (%s) after launching synthetiseImageKernel!\n", cudaStatus, cudaGetErrorString(cudaStatus));
				goto Error;
			}

			cudaStatus = cudaMemcpy(pontuationsArray, cuda_pontuationsArray, pontuationsArraySize * sizeof(float), cudaMemcpyDeviceToHost);
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaMemcpy pontuationsArray failed!");
				goto Error;
			}

			int winnerPontuationItt = 0;
			float winnerPontuation = std::numeric_limits<float>::max();
			for (int pontuationItt = 0; pontuationItt < pontuationsArraySize; pontuationItt++) {
				if (pontuationsArray[pontuationItt] < winnerPontuation) {
					winnerPontuation = pontuationsArray[pontuationItt];
					winnerPontuationItt = pontuationItt;
				}
			}

			int winnerPixelPos = winnerPontuationItt * sampleChannels;

			int pixelPos = x * preSynChannels + (y * preSynWidth * preSynChannels);
			synthetizedData[pixelPos + 0] = sampleData[winnerPixelPos + 0];
			synthetizedData[pixelPos + 1] = sampleData[winnerPixelPos + 1];
			synthetizedData[pixelPos + 2] = sampleData[winnerPixelPos + 2];

			cudaStatus = cudaMemcpy(cuda_synthetizedData, synthetizedData, preSynSizeBytes, cudaMemcpyHostToDevice);
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaMemcpy failed!");
				goto Error;
			}
		}
	}

	// Copy synthetized data back
	cudaStatus = cudaMemcpy(synthetizedData, cuda_synthetizedData, preSynSizeBytes, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(cuda_synthetizedData);
	cudaFree(cuda_sampleData);

	return cudaStatus;
}