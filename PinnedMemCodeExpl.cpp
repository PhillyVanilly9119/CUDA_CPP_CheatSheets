#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
using namespace std;

float timeMemory(bool pinned, bool toDevice) 
{
	const int count = 1 << 20; //2^20
	const int interations = 1 << 6;
	const int size = count * sizeof(int); //size of memory

	cudaEvent_t start, end;
	int *h, *d; //host and devive memory
	float elapsed;
	cudaError_t status;

	cudaEventCreate(&start);
	cudaEventCreate(&end);

	cudaMalloc(&d, size); // allocate device memory
	if (pinned)
		cudaHostAlloc(&h, size, cudaHostAllocDefault);
	else
		h = new int[count];

	cudaEventRecord(start);

	for (int i = 0; i < interations; i++) 
	{
		if (toDevice)
			status = cudaMemcpy(d, h, size, cudaMemcpyHostToDevice);
		else 
			status = cudaMemcpy(h, d, size, cudaMemcpyHostToDevice);
	}

	cudaEventRecord(end);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&elapsed, start, end);

	if (pinned)
		cudaFreeHost(h);
	else
		delete[] h;

	cudaFree(d);
	cudaEventDestroy(start);
	cudaEventDestroy(end);

	return elapsed;
}

int main()
{
	cout << "From device,paged memory:\t" << timeMemory(false, false) << endl;
	cout << "TO device,	paged memory:\t" << timeMemory(false, false) << endl;
	cout << "From device, pinned memory:\t" << timeMemory(false, false) << endl;
	cout << "To device, pinned memory:\t" << timeMemory(false, false) << endl;

	getchar();
	return 0;
}