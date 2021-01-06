#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#define BLOCK_SIZE 16
__global__ void mandelKernel(int* device_img, float lowerX, float lowerY, float stepX, float stepY, int width, int height, int maxIterations) 
{
    // To avoid error caused by the floating number, use the following pseudo code
    // float x = lowerX + thisX * stepX;
    // float y = lowerY + thisY * stepY;

    unsigned int thisX = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int thisY = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (thisX < width && thisY < height) {
        int idx = thisY * width + thisX;
        float c_re = lowerX + thisX * stepX;
        float c_im = lowerY + thisY * stepY;
        float z_re = c_re, z_im = c_im;
        int i = 0;
        for (i = 0; i < maxIterations; ++i)
        {
            if (z_re * z_re + z_im * z_im > 4.f)
                break;

            float new_re = z_re * z_re - z_im * z_im;
            float new_im = 2.f * z_re * z_im;
            z_re = c_re + new_re;
            z_im = c_im + new_im;
        }
        device_img[idx] = i;
    }
}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE (float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations)
{
    int* device_img, *host_img;
    float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;
    cudaMalloc((void **)&device_img, resX * resY * sizeof(int));
    host_img = (int *) malloc(resX * resY * sizeof(int)); 

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlock(resX / BLOCK_SIZE, resY / BLOCK_SIZE);
    mandelKernel<<<numBlock, blockSize>>>(device_img, lowerX, lowerY, stepX, stepY, resX, resY, maxIterations);
    
    cudaDeviceSynchronize();
    cudaMemcpy(host_img, device_img, resX * resY * sizeof(int), cudaMemcpyDeviceToHost);
    memcpy(img,host_img,resX * resY * sizeof(int));
    cudaFree(device_img);
}
