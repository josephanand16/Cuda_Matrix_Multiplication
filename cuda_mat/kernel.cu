#include "cuda_runtime.h"

#include<iostream>
#include <chrono>
#include <cstdlib>
#include "device_launch_parameters.h"

// Select the size of the matix of dim [ SIZE X SIZE ]
#define SIZE 1024

using namespace std;

//Ensure to add the __global__ block when performing GPU operations
__global__  void gpu_multer(double *A, double *B, double *C)
{
    // ID and dimensions need to be computed to run the particular matrix multiplication operation
    // On a cuda core
    // The 2D SIZE Matrix is converted into a SIZE * SIZE linear array
    // i and j takes the row and column indices
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    //*C = 0.0;
    if (i < SIZE && j < SIZE)
    {
        for (int k = 0; k < SIZE; k++) {
            // Row and column indices are transformed to linear
            C[j * SIZE + i] += A[j * SIZE + k] * B[k * SIZE + i];
        }
    }
        
}

// Functions for CPU operation
void cpu_multer(double* A,double* B, double* C_cpu)
{

    for (int i = 0; i < SIZE; i++)
    {
        for (int j = 0; j < SIZE; j++)
        {
            for (int k = 0; k < SIZE; k++)
            {
                //The same as GPU is considered for CPU Matrix Multiplication
                C_cpu[i * SIZE + j] += (A[i * SIZE + k] * B[k * SIZE + j]);
            }
            //cout << C_cpu[i * SIZE + j] << "  ";
        }
        //cout << "\n";
    }
}

// Function to print the matrix
void printMat(double *A)
{
    for (int i = 0; i < SIZE; i++)
    {
        for (int j = 0; j < SIZE; j++)
        {
            cout << A[i * SIZE + j] << "  ";
        }
        cout << "\n";
    }
}

int main()

{
    using milli = std::chrono::milliseconds;
    using micro = std::chrono::microseconds;

    // Define address in CPU space
    double* A, * B, * C;
    // Define address in GPU space
    // cudaMallocManaged copies variable to GPU
    cudaMallocManaged(&A, SIZE * SIZE * sizeof(double));
    cudaMallocManaged(&B, SIZE * SIZE * sizeof(double));
    cudaMallocManaged(&C, SIZE * SIZE * sizeof(double));

    double* C_cpu = (double*)malloc(SIZE * SIZE * sizeof(double));
    for (int i = 0; i < SIZE; i++)
    {
        for (int j = 0; j < SIZE; j++)
        {
            for (int k = 0; k < SIZE; k++)
            {
                C_cpu[i * SIZE + j] = 0;
            }
        }
    }

    //Random double precision data is populated Array A
    for (int i = 0; i < SIZE; i++)
    {
        for (int j = 0; j < SIZE; j++)
        {
            A[i * SIZE + j] = (double) (rand() % 10000000) / 1000000;
        }
    }

    //Random double precision data is populated Array B
    for (int i = 0; i < SIZE; i++)
    {
        for (int j = 0; j < SIZE; j++)
        {
            B[i * SIZE + j] = (double)(rand() % 10000000) / 1000000;
        }
    }
  
    // GPU multiplication operation along with time
    cout << "GPU Computation start\n";
    auto start = std::chrono::high_resolution_clock::now();
    gpu_multer <<< dim3(SIZE, SIZE), dim3(SIZE, SIZE) >>> (A, B, C);
    cudaDeviceSynchronize();
    auto finish = std::chrono::high_resolution_clock::now();

    cout << "Multiplication of matrices in GPU\n";
    cout << "Matrix mulitplication took "
        << std::chrono::duration_cast<micro>(finish - start).count()
        << " microseconds\n\n";

    // CPU Multiplication operation
    cout << "CPU Computation start\n";
    auto start1 = std::chrono::high_resolution_clock::now();
    cpu_multer(A, B, C_cpu);
    auto finish1 = std::chrono::high_resolution_clock::now();

    cout << "Multiplication of matrices in CPU\n";
    cout << "Matrix mulitplication took "
        << std::chrono::duration_cast<milli>(finish1 - start1).count()
        << " milliseconds\n\n";

    // Use this block for debugging
    // Checking the array ouyput

    //cout << "> A :\n";
    //printMat(A);
    //cout << "\n";
    //cout << "> B :\n";
    //printMat(B);
    //cout << "\n";
    //cout << "> CPU computed Matrix :\n";
    //printMat(C_cpu);
    //cout << "\n";
    //cout << "> GPU computed Matrix :\n";
    //printMat(C);
    //cout << "\n";

    // Always free allocated memory after completion of execution
    // cudaFree is a custom cuda api
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    free(C_cpu);

    return 0;

}