
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#define BLOCK_SIZE 25
__global__ void mandelKernel(
        int* device_img, float lowerX, float lowerY, float stepX,
        float stepY, int width, int height, int maxIterations, 
        int g_width, int g_height, size_t pitch) {
    // To avoid error caused by the floating number, use the following pseudo code
    // float x = lowerX + thisX * stepX;
    // float y = lowerY + thisY * stepY;

    unsigned int thisX = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int thisY = blockIdx.y * blockDim.y + threadIdx.y;
    int final_y = thisY + g_height;
    int final_x = thisX + g_width;
    for (int j = thisY; j < final_y; j++)
    {
        for (int i = thisX; i < final_x; i++)
        {
            if (i < width && j < height) {
                int idx = j * width + i;
                float c_re = lowerX + i * stepX;
                float c_im = lowerY + j * stepY;
                float z_re = c_re, z_im = c_im;
                int val = 0;
                for (val = 0; val < maxIterations; val++)
                {
                    if (z_re * z_re + z_im * z_im > 4.f)
                        break;
                    float new_re = z_re * z_re - z_im * z_im;
                    float new_im = 2.f * z_re * z_im;
                    z_re = c_re + new_re;
                    z_im = c_im + new_im;
                }
                device_img[idx] = val;
            }
        }
    }
}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE (float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations)
{
    int* device_img, *host_img;
    float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;

    size_t pitch; 
    cudaMallocPitch((void **)&device_img, &pitch, sizeof(float)*resX, resY); 
    cudaHostAlloc((void **)&host_img, resX * resY * sizeof(int),cudaHostAllocDefault);
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlock(resX / BLOCK_SIZE, resY / BLOCK_SIZE);
    int g_width = numBlock.x / BLOCK_SIZE;
    int g_height = numBlock.y / BLOCK_SIZE;
 
    mandelKernel<<<numBlock, blockSize>>>(device_img, lowerX, lowerY, stepX, stepY, resX, resY, maxIterations, g_width, g_height, pitch);

    cudaDeviceSynchronize();
    cudaMemcpy(host_img, device_img, resX * resY * sizeof(int), cudaMemcpyDeviceToHost);
    memcpy(img, host_img,resX * resY * sizeof(int));
    cudaFree(device_img);
}
