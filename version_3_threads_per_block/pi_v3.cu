
#include <iostream>
#include <limits>
#include <cuda.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <chrono>
#include <fstream>
#include <random>
#include <cstring>
#include <cuda_runtime.h>

using namespace std;

typedef unsigned long long Count;

const int seq_iter = 100000 ;

const Count N_BLOCKS = 1024;
const Count N_THREADS = 1024;
const Count WARP_SIZE = 32;

// This kernel is 
__global__ void picount(Count *totals, int seq_iter) {
	// Define some shared memory: all threads in this block
	__shared__ Count counter[N_THREADS];

	// Unique ID of the thread
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	// Initialize RNG
	curandState_t rng;
	curand_init(clock64(), tid, 0, &rng);

	// Initialize the counter
	counter[threadIdx.x] = 0;

	// Computation loop
	for (int i = 0; i < seq_iter; i++) {
		float x = curand_uniform(&rng); // Random x position in [0,1]
		float y = curand_uniform(&rng); // Random y position in [0,1]
		counter[threadIdx.x] += 1 - int(x * x + y * y); // Hit test
	}

	__syncthreads();

	// The first thread in *every block* should sum the results
	if (threadIdx.x == 0) {
		// Reset count for this block
		totals[blockIdx.x] = 0;
		// Accumulate results
		for (int i = 0; i < N_THREADS; i++) {
			totals[blockIdx.x] += counter[i];
		}
	}
}

int main(int argc, char **argv) {


	// Allocate host and device memory to store the counters
	Count *hOut, *dOut;
	hOut = new Count[N_BLOCKS]; // Host memory
	cudaMalloc(&dOut, sizeof(Count)*N_BLOCKS); // Device memory

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record the start time
    cudaEventRecord(start, 0);


	// Launch kernel
	picount<<<N_BLOCKS,N_THREADS>>>(dOut,seq_iter);


    // Record the stop time
    cudaEventRecord(stop, 0);

    // Synchronize and measure the elapsed time
    cudaEventSynchronize(stop);
    float kernel_time_ms;
    cudaEventElapsedTime(&kernel_time_ms, start, stop);

    std::cout << "Kernel execution time: " << kernel_time_ms << " ms" << std::endl;

    // Clean up CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);


	// Copy back memory used on device and free
	cudaMemcpy(hOut, dOut, sizeof(Count)*N_BLOCKS,cudaMemcpyDeviceToHost);
	cudaFree(dOut);

	// Compute total hits
	Count total = 0;
	for(int i=0; i<N_BLOCKS; i++)
	{
		total += hOut[i];
	}

	Count tests = seq_iter*N_BLOCKS*N_THREADS;
	cout << "Approximated PI using " << tests << " random tests\n";

	// Set maximum precision for decimal printing
	cout.precision(std::numeric_limits<double>::max_digits10);
	cout << "PI ~= " << 4.0 * (double)total/(double)tests << endl;
	cout << "Tests per ms: " << tests/kernel_time_ms << endl;


	return 0;
}
