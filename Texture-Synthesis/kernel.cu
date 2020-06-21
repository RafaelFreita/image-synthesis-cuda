
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iterator>
#include <limits>

#include <stb_image.h>
#include "stb_image_write.h"

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

#pragma region CPU

unsigned char* synthetiseImage(Image& sampleImage, Image& preSynImage, uint8_t neighborhoodSize) {

	if (neighborhoodSize % 2 == 0) {
		printf("ERROR neighborhoodSize for image synthesis must be odd!\n");
		return nullptr;
	}

	int arrSize = preSynImage.width * preSynImage.height * 3;
	unsigned char* newImage = new unsigned char[arrSize];

	for (int i = 0; i < arrSize; i++)
	{
		newImage[i] = preSynImage.data[i];
	}

	int imageTempX = 0, imageTempY = 0;
	int tempX = 0, tempY = 0;

	uint16_t winnerPixelPosX, winnerPixelPosY;
	float winnerPontuation;
	float pontuation = 0;

	uint8_t neighborhoodHalf = (neighborhoodSize / 2);

	// For every pixel
	for (int y = 0; y < preSynImage.height; y++)
	{
		for (int x = 0; x < preSynImage.width; x++)
		{

			winnerPontuation = INT_MAX;

			// For every pixel in sample image
			for (int sY = 0; sY < sampleImage.height; sY++)
			{
				for (int sX = 0; sX < sampleImage.width; sX++)
				{

					pontuation = 0.0f;

					// Check neighbors - From upper corner to the pixel itself
					for (int nY = -neighborhoodHalf; nY <= 0; nY++)
					{

						imageTempY = y + nY;
						if (imageTempY < 0) {
							imageTempY = preSynImage.height + imageTempY;
						}
						else if (imageTempY >= preSynImage.height) {
							imageTempY = abs(preSynImage.height - imageTempY);
						}

						tempY = sY + nY;
						if (tempY < 0) {
							tempY = sampleImage.height + tempY;
						}
						else if (tempY >= sampleImage.height) {
							tempY = abs(sampleImage.height - tempY);
						}

						for (int nX = -neighborhoodHalf; nX <= neighborhoodHalf; nX++)
						{
							if (nY == 0 && nX > 0) { // Last line going only until the proper pixel
								break;
							}

							imageTempX = x + nX;
							if (imageTempX < 0) {
								imageTempX = preSynImage.width + imageTempX;
							}
							else if (imageTempX >= preSynImage.width) {
								imageTempX = abs(preSynImage.width - imageTempX);
							}

							tempX = sX + nX;
							if (tempX < 0) {
								tempX = sampleImage.width + tempX;
							}
							else if (tempX >= sampleImage.width) {
								tempX = abs(sampleImage.width - tempX);
							}

							// Compare pixels from sample with preSyn
							// 0 - R // 1 - G // 2 - B
							// Multiplying for 3 because every pixel has 3 components

							int newImagePos = imageTempX * preSynImage.channelsQtd + (imageTempY * preSynImage.width * preSynImage.channelsQtd);
							int tempPos = tempX * sampleImage.channelsQtd + (tempY * sampleImage.width * sampleImage.channelsQtd);

							float dist = distanceBetweenColors(
								newImage[newImagePos + 0],
								newImage[newImagePos + 1],
								newImage[newImagePos + 2],

								sampleImage.data[tempPos + 0],
								sampleImage.data[tempPos + 1],
								sampleImage.data[tempPos + 2]
							);

							pontuation += dist;
						}
					}
					// End of neighbors

					// Assign new winner if pontution was less
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
			newImage[pixelPos + 0] = sampleImage.data[winerPixelPos + 0];
			newImage[pixelPos + 1] = sampleImage.data[winerPixelPos + 1];
			newImage[pixelPos + 2] = sampleImage.data[winerPixelPos + 2];

		}
	}

	return newImage;
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

	int32_t imageTempX = 0, imageTempY = 0;
	int32_t tempX = 0, tempY = 0;

	float pontuation = 0;

	int16_t sX = threadIdx.x + blockIdx.x * blockDim.x;
	int16_t sY = threadIdx.y + blockIdx.y * blockDim.y;

	for (int nY = -neighborhoodHalf; nY <= 0; nY++)
	{

		imageTempY = y + nY;
		if (imageTempY < 0) imageTempY = preSynHeight + imageTempY;
		else if (imageTempY >= preSynHeight) imageTempY = abs(preSynHeight - imageTempY);

		tempY = sY + nY;
		if (tempY < 0) tempY = sampleHeight + tempY;
		else if (tempY >= sampleHeight) tempY = abs(sampleHeight - tempY);

		for (int nX = -neighborhoodHalf; nX <= neighborhoodHalf; nX++)
		{
			if (nY == 0 && nX > 0) { // Last line going only until the proper pixel
				break;
			}

			imageTempX = x + nX;
			if (imageTempX < 0) imageTempX = preSynWidth + imageTempX;
			else if (imageTempX >= preSynWidth) imageTempX = abs(preSynWidth - imageTempX);

			tempX = sX + nX;
			if (tempX < 0) tempX = sampleWidth + tempX;
			else if (tempX >= sampleWidth) tempX = abs(sampleWidth - tempX);

			// Compare pixels from sample with preSyn
			// 0 - R // 1 - G // 2 - B

			int newImagePos = imageTempX * preSynChannels + (imageTempY * preSynWidth * preSynChannels);
			int tempPos = tempX * sampleChannels + (tempY * sampleWidth * sampleChannels);

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

int main()
{
	//
	// Loading images
	//
	stbi_set_flip_vertically_on_load(true);

	Image sampleImg;
	sampleImg.Load("images/flowers256.png");

	Image preSynImg;
	preSynImg.Load("images/preSynFlowers256.png");

	uint8_t neighborhoodSize = 5;
	/*unsigned char* synData = synthetiseImage(sampleImg, preSynImg, neighborhoodSize);
	if (synData == nullptr) {
		system("pause");
		return 1;
	}*/

	Image synthetizedImage;
	synthetizedImage.width = preSynImg.width;
	synthetizedImage.height = preSynImg.height;
	synthetizedImage.channelsQtd = 3;
	synthetizedImage.AllocateDataArray();

#pragma region Cuda calls
	cudaError_t cudaStatus = synthetiseImageCuda(
		sampleImg.width, sampleImg.height, sampleImg.channelsQtd, sampleImg.data,
		preSynImg.width, preSynImg.height, preSynImg.channelsQtd, preSynImg.data,
		5, synthetizedImage.data, 16U
	);
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
#pragma endregion

	synthetizedImage.WriteImage("images/resultFlowers256.png");

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

	cudaError_t cudaStatus;

	uint8_t neighborhoodHalf = neighborhood / 2;

#pragma region Cuda Stuff

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
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

	cudaStatus = cudaMemcpy(cuda_sampleData, sampleData, sampleSizeBytes, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(cuda_synthetizedData, synthetizedData, preSynSizeBytes, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

#pragma endregion

	dim3 threadsPerBlockDim(threadsPerBlock, threadsPerBlock);
	dim3 numBlocks(sampleWidth / threadsPerBlockDim.x, sampleHeight / threadsPerBlockDim.y);

	for (int y = 0; y < preSynHeight; y++)
	{
		for (int x = 0; x < preSynWidth; x++)
		{

			synthetiseImageKernel << <numBlocks, threadsPerBlockDim >> > (
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