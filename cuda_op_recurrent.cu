#define EIGEN_USE_GPU
#include <cuda.h>
#include <stdio.h>
#include <algorithm>


#define THREADS_PER_BLOCK 1024
#define MAX_STREAMS 100
using namespace std; 


__global__ void getValidKernalPositions(
    const int block,
    const int input_feature_width,
    const int kernal_width,
    const int *kernal_deltas,
    const int *kernal_postions,
    int *kernel_pointers,
    int threadOffset = 0) 
{
    const int threadID = threadOffset + threadIdx.x;
    const int ix = block * (input_feature_width*input_feature_width) + (threadID / (kernal_width*kernal_width)); 
    const int kernal_start = block     *  (input_feature_width*input_feature_width);
    const int kernal_end   = (block+1) *  (input_feature_width*input_feature_width);

    const int row = (threadID % (kernal_width*kernal_width)) / kernal_width;
    const int row_iX = ix + kernal_postions[(row * kernal_width) + int(kernal_width/2)];

    const int column = (threadID % (kernal_width*kernal_width)) % kernal_width;

    if ( (kernal_start <= row_iX) && (kernal_end > row_iX) )
    {
        const int row_start = (row_iX < input_feature_width) ? 0 : (row_iX - ( (row_iX - input_feature_width) % input_feature_width ));
        const int row_end = row_start + input_feature_width;

        const int p         = ((row * kernal_width) + column);
        const int pointer   = (ix * ( kernal_width*kernal_width ) ) + p;
        const int location  = ix + kernal_postions[p];

        if ( (row_start <= location) && (row_end > location) )
        {
            kernel_pointers[pointer] = location;
        } else {
            kernel_pointers[pointer] = -1;
        }
    } else {
        const int pointer = (ix * ( kernal_width*kernal_width ) ) + ((row * kernal_width) + column);
        kernel_pointers[pointer] = -1;
    }
}

__global__ void RecurrentKernel_splitThreaded(
    const int block,
    const double* inputs,
    const double* weights,
    const int iterations,
    const int batch_samples, 
    const int units, 
    const int input_feature_width,
    const int kernal_width,
    const int *kernal_deltas,
    int *kernel_pointers,
    double *output,
    double *output_clones) 
{
    const int cell = (block * units) + threadIdx.x;

    for (int i = 0; i < iterations; i++)
    {
         __syncthreads();

         for (int j = 0; j < (kernal_width*kernal_width); j++)
         {
             const int cell_pointer      = (cell * (kernal_width*kernal_width))  + j;
             const int cell_position = kernel_pointers[cell_pointer];

            if (cell_position >= 0.0)
            {   
                const int weight_pointer = (block * (kernal_width*kernal_width)) + j;
                const double weight = weights[ weight_pointer ]/100;
                const double value  = output[cell];
                const double o = value * weight;
                atomicAdd(&output_clones[ cell_position ], o);
            }
        }
        __syncthreads();

        output[cell] = output_clones[cell];
    }

    // RELU
    output[cell] = (output[cell] > 0) ? output[cell] : 0;
}

__global__ void RecurrentKernel(
    const int block,
    const double* inputs,
    const double* weights,
    const int iterations,
    const int batch_samples, 
    const int units, 
    const int input_feature_width,
    const int kernal_width,
    const int *kernal_deltas,
    int *kernel_pointers,
    double *output,
    double *output_clones) 
{
    const int cell = block * (input_feature_width*input_feature_width) + (threadIdx.x / (kernal_width*kernal_width)); 
    const int offset = (threadIdx.x % (kernal_width*kernal_width));
    const int cell_pointer      = (cell * (kernal_width*kernal_width)) + offset;
    const int cell_position = kernel_pointers[cell_pointer];

    for (int i = 0; i < iterations; i++)
    {
         __syncthreads();
        if (cell_position >= 0.0)
        {   
            const int weight_pointer = (block * (kernal_width*kernal_width)) + offset; 
            const double weight = weights[weight_pointer]/100;
            const double value  = output[cell];
            const double o = value * weight;
            atomicAdd(&output_clones[cell_position], o);
        }

        __syncthreads();

        output[cell] = output_clones[cell];
    }

    // RELU
    output[cell] = (output[cell] > 0) ? output[cell] : 0;
}

void RecurrentKernelLauncher(
        const double* inputs, 
        const double* weights,
        const int iterations,
        const int batch_samples, 
        const int units, 
        const int input_feature_width,
        const int kernal_width,
        double* output) 
{
    const int    numOfThreads = (units*(kernal_width*kernal_width));
    const size_t inputBytes = (units * batch_samples) * sizeof(double);
    const size_t kernalCellPositionBytes = (( units * batch_samples ) * (kernal_width * kernal_width )) * sizeof(int);
    const size_t kernalBytes = ( kernal_width * kernal_width ) * sizeof(int);
    const size_t weightBytes = (( kernal_width * kernal_width ) * batch_samples) * sizeof(double);
    size_t sharedMemory = (inputBytes*3) + kernalCellPositionBytes + kernalBytes + weightBytes;
    printf("sharedMemory: %i bytes\n", sharedMemory);

    // blocks are limited to 1024 // if threads go beyond this, threads are broken up by row
    int threadSplit     = 0;
    int threadIntervals = 0;
    int endingThread    = 0;
    int numOfStreams    = 0;
    if (numOfThreads > THREADS_PER_BLOCK) {
        threadSplit = (int)(numOfThreads/THREADS_PER_BLOCK)+1;
        threadIntervals = numOfThreads/threadSplit;
        endingThread = numOfThreads - ((threadSplit-1) * threadIntervals);
    }
    if (batch_samples > MAX_STREAMS) {
        numOfStreams = MAX_STREAMS;
    } else {
        numOfStreams = batch_samples;
    }

    // allocate and initialize an array of stream handles
    cudaStream_t *streams;
    int *kernal_deltas, *d_kernal_deltas, *kernel_pointers, *d_kernel_pointers; 
    double *d_output, *d_output_clones, *d_inputs, *d_weights;

    cudaMallocHost(&streams,(numOfStreams * sizeof(cudaStream_t)));
    for (int i = 0; i < numOfStreams; i++) cudaStreamCreate(&(streams[i]));

    cudaMallocHost(&kernal_deltas, kernalBytes);

    int strtPos = -(kernal_width/2);
    int endPos  =   kernal_width - abs(strtPos);
    int pointer = 0;

    for(int fx=strtPos; fx < endPos; fx++) for(int fy=strtPos; fy < endPos; fy++)   {
        kernal_deltas[pointer] = ((fx * input_feature_width) + fy);
        pointer++;
    }

    cudaMalloc( (void **)&d_kernal_deltas, kernalBytes );
    cudaMemcpy(d_kernal_deltas, kernal_deltas, kernalBytes, cudaMemcpyHostToDevice);

    cudaMalloc( (void **)&d_output, inputBytes );
    cudaMemcpy(d_output, inputs, inputBytes, cudaMemcpyHostToDevice);

    cudaMalloc( (void **)&d_output_clones, inputBytes );
    cudaMemcpy(d_output_clones, inputs, inputBytes, cudaMemcpyHostToDevice);

    cudaMalloc( (void **)&d_inputs, inputBytes );
    cudaMemcpy(d_inputs, inputs, inputBytes, cudaMemcpyHostToDevice);

    cudaMalloc( (void **)&d_weights, weightBytes );
    cudaMemcpy(d_weights, weights, weightBytes, cudaMemcpyHostToDevice);

    cudaMallocHost( (void **)&kernel_pointers, kernalCellPositionBytes );
    cudaMalloc( (void **)&d_kernel_pointers, kernalCellPositionBytes );
    cudaMemcpy(d_kernel_pointers, kernel_pointers, kernalCellPositionBytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    int stream = 0;
    if (numOfThreads > THREADS_PER_BLOCK)
    {
        for (int i = 0; i < batch_samples; i++) {
            for (int j = 0; j < threadSplit; j++) {
                int k_threads = (j == ( threadSplit-1 )) ? endingThread : threadIntervals;
                getValidKernalPositions<<<1, k_threads, 0, streams[stream]>>>(i, input_feature_width, kernal_width, kernal_deltas,d_kernal_deltas, d_kernel_pointers, j*threadIntervals);
                stream++;
                if ((stream % numOfStreams) == 0) stream = 0;
            }
        }
    } else {
        for (int i = 0; i < batch_samples; i++) {
            getValidKernalPositions<<<1, numOfThreads, 0, streams[stream]>>>(i, input_feature_width, kernal_width, kernal_deltas,d_kernal_deltas, d_kernel_pointers);
            stream++;
            if ((stream % numOfStreams) == 0) stream = 0;
        }
    }
    cudaDeviceSynchronize();

    cudaMemcpy(kernal_deltas, d_kernal_deltas, kernalBytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(d_kernal_deltas, kernal_deltas, kernalBytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    // batch_samples is the number of instances of the kernel
    // units is the number of threads within each instance
    stream = 0;
    if (numOfThreads > THREADS_PER_BLOCK)
    {
        for (int i = 0; i < batch_samples; i++) {
            RecurrentKernel_splitThreaded<<<1, units, 0, streams[stream]>>>(i, d_inputs, d_weights,iterations, batch_samples, units, input_feature_width,kernal_width, kernal_deltas, d_kernel_pointers, d_output, d_output_clones);
            stream++;
            if ((stream % numOfStreams) == 0) stream = 0;
        }
    } else {
        for (int i = 0; i < batch_samples; i++) {
            RecurrentKernel<<<1, numOfThreads, 0, streams[stream]>>>(i, d_inputs, d_weights, iterations, batch_samples, units, input_feature_width,kernal_width, kernal_deltas, d_kernel_pointers, d_output, d_output_clones);
            stream++;
            if ((stream % numOfStreams) == 0) stream = 0;
        }
    }

    cudaError_t cudaerr = cudaDeviceSynchronize();
    cudaMemcpy(output, d_output, inputBytes, cudaMemcpyDeviceToHost);

    cudaFreeHost(kernal_deltas);
    cudaFreeHost(kernel_pointers);
    cudaFree(d_kernal_deltas);
    cudaFree(d_kernel_pointers);
    cudaFree(d_output);
    cudaFree(d_output_clones);
    cudaFree(d_inputs);
    cudaFree(d_weights);
    if (cudaerr != cudaSuccess)
    {
        printf("kernel launch failed with error \"%s\".\n", cudaGetErrorString(cudaerr));
    }
}