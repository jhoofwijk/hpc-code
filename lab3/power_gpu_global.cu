// the subroutine for GPU code can be found in several separated text file from the Brightspace. 
// You can add these subroutines to this main code.
////////////////////////////////////////////


#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "cuda.h"

const int BLOCK_SIZE =32;  // number of threads per block

// Input Array Variables
float* h_MatA = NULL;
float* d_MatA = NULL;

// Output Array
float* h_VecV = NULL;
float* d_VecV = NULL;
float* h_VecW = NULL;
float* d_VecW = NULL;
float* h_NormW = NULL;
float* d_NormW = NULL;

// Variables to change
int GlobalSize = 5000;         // this is the dimension of the matrix, GlobalSize*GlobalSize
int BlockSize = 32;            // number of threads in each block
// const float EPS = 0.000005;    // tolerence of the error
// int max_iteration = 100;       // the maximum iteration steps
const float EPS = -1;
int max_iteration = 10;


// Functions
void Cleanup(void);
void InitOne(float*, int);
void UploadArray(float*, int);
float CPUReduce(float*, int);
void ParseArguments(int, char**);
void checkCardVersion(void);

// Kernels
__global__ void Av_Product(float* g_MatA, float* g_VecV, float* g_VecW, int N);
__global__ void FindNormW(float* g_VecW, float* g_tempV, float * g_NormW, int N);
__global__ void NormalizeW(float* g_VecW, float * g_NormW, float* g_VecV, int N);
__global__ void ComputeLamda( float* g_VecV, float* g_VecW, float * g_Lamda,int N);


void CPU_AvProduct()
{
	int N = GlobalSize;
	int matIndex =0;
    for(int i=0;i<N;i++)
	{
		h_VecW[i] = 0;
		for(int j=0;j<N;j++)
		{
			matIndex = i*N + j;
			h_VecW[i] += h_MatA[matIndex] * h_VecV[j];
		}
	}
}

void CPU_NormalizeW()
{
	int N = GlobalSize;
	float normW=0;
	for(int i=0;i<N;i++)
		normW += h_VecW[i] * h_VecW[i];
	
	normW = sqrt(normW);
	for(int i=0;i<N;i++)
		h_VecV[i] = h_VecW[i]/normW;
}

float CPU_ComputeLamda()
{
	int N = GlobalSize;
	float lamda =0;
	for(int i=0;i<N;i++)
		lamda += h_VecV[i] * h_VecW[i];
	
	return lamda;
}

void RunCPUPowerMethod()
{
	printf("*************************************\n");
	float oldLamda =0;
	float lamda=0;
	
	//AvProduct
	CPU_AvProduct();
	
	//power loop
	for (int i=0;i<max_iteration;i++)
	{
        CPU_NormalizeW();
        // for(int j=0;j<10;j++) {
        //     printf("%.5f   ", h_VecV[i]);
        // }
        // printf("\n");

		CPU_AvProduct();
		lamda= CPU_ComputeLamda();
		printf("CPU lamda at %d: %f \n", i, lamda);
		// If residual is lass than epsilon break
		if(abs(oldLamda - lamda) < EPS)
			break;
		oldLamda = lamda;	
	
	}
	printf("*************************************\n");
	
}

void start_timer(struct timespec *timer) {
    clock_gettime(CLOCK_REALTIME, timer);  // Here I start to count
}

double stop_timer(struct timespec *timer) {
    struct timespec t_end;

    clock_gettime(CLOCK_REALTIME, &t_end);
    return (t_end.tv_sec - timer->tv_sec) + 1e-9*(t_end.tv_nsec - timer->tv_nsec);
}


// Host code
int main(int argc, char** argv)
{
    srand(time(0));
    struct timespec timer;
    double runtime;
    ParseArguments(argc, argv);
		
    int N = GlobalSize;
    printf("Matrix size %d X %d \n", N, N);
    printf("BlockSize: %d\n\n", BLOCK_SIZE);

    size_t vec_size = N * sizeof(float);
    size_t mat_size = N * N * sizeof(float);
    size_t norm_size = sizeof(float);
  
    // Allocate normalized value in host memory
    h_NormW = (float*)malloc(norm_size);
    // Allocate input matrix in host memory
    h_MatA = (float*)malloc(mat_size);
    // Allocate initial vector V in host memory
    h_VecV = (float*)malloc(vec_size);
    // Allocate W vector for computations
    h_VecW = (float*)malloc(vec_size);


    // Initialize input matrix
    UploadArray(h_MatA, N);
    InitOne(h_VecV,N);

    printf("Power method in CPU starts\n");	   
    
    start_timer(&timer);
    RunCPUPowerMethod();   // the lamda is already solved here
    runtime = stop_timer(&timer);

    printf("CPU: run time = %f secs.\n",runtime);
    printf("Power method in CPU is finished\n\n\n");
    
    
    /////////////////////////////////////////////////
    // This is the starting points of GPU
    printf("Power method in GPU starts\n");
    checkCardVersion();

    // Initialize input matrix
    InitOne(h_VecV,N);
    
    start_timer(&timer);  // Here I start to count
    double gpu_time=0.0;

    // Set the kernel arguments
    int threadsPerBlock = BlockSize;   
    int sharedMemSize = threadsPerBlock * threadsPerBlock * sizeof(float); // in per block, the memory is shared   
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Allocate matrix and vectors in device memory
    cudaMalloc((void**)&d_MatA, mat_size); 
    cudaMalloc((void**)&d_VecV, vec_size); 
    cudaMalloc((void**)&d_VecW, vec_size); // This vector is only used by the device
    cudaMalloc((void**)&d_NormW, norm_size); 

    //Copy from host memory to device memory
    cudaMemcpy(d_MatA, h_MatA, mat_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_VecV, h_VecV, vec_size, cudaMemcpyHostToDevice);
	// cutilCheckError(cutStopTimer(timer_mem));
	  
    //Power method loops
    printf("*************************************\n");
	float lamda = 0;
    float OldLamda =0;
    
    // start_timer(&timer);
    Av_Product<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_MatA, d_VecV, d_VecW, N);
    cudaThreadSynchronize(); //Needed, kind of barrier to sychronize all threads
    // gpu_time += stop_timer(&timer);
	
    // This part is the main code of the iteration process for the Power Method in GPU. 
    // Please finish this part based on the given code. Do not forget the command line 
    // cudaThreadSynchronize() after calling the function every time in CUDA to synchoronize the threads
	
    //power loop
    float zero = 0.0;
	for (int i=0;i<max_iteration;i++)
	{
        cudaMemcpy(d_NormW, &zero, sizeof(float), cudaMemcpyHostToDevice);
        // start_timer(&timer);
        FindNormW<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(d_VecW, d_VecV, d_NormW, N);
        // gpu_time += stop_timer(&timer);
        

        cudaMemcpy(h_NormW, d_NormW, sizeof(float), cudaMemcpyDeviceToHost);
        h_NormW[0] = sqrt(h_NormW[0]);        
        cudaMemcpy(d_NormW, h_NormW, sizeof(float), cudaMemcpyHostToDevice);
        


        // start_timer(&timer);
        NormalizeW<<<blocksPerGrid, threadsPerBlock, sizeof(float)>>>(d_VecW, d_NormW, d_VecV, N);
        
        Av_Product<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_MatA, d_VecV, d_VecW, N);
        // gpu_time += stop_timer(&timer);
        
        
        
        cudaMemcpy(d_NormW, &zero, sizeof(float), cudaMemcpyHostToDevice);
        // start_timer(&timer);
        ComputeLamda<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(d_VecV, d_VecW, d_NormW, N);
        // gpu_time += stop_timer(&timer);
        cudaMemcpy(&lamda, d_NormW, sizeof(float), cudaMemcpyDeviceToHost);
        
        cudaThreadSynchronize();
        printf("GPU lamda at %d: %f \n", i, lamda);
		// If residual is less than epsilon break
		if(abs(OldLamda - lamda) < EPS)
            break;
        OldLamda = lamda;	    
	}
	printf("*************************************\n");
    gpu_time = stop_timer(&timer);
    printf("GPU: run time = %f secs.\n\n\n\n\n",gpu_time);
    // printf("Overall CPU Execution Time: %f (ms) \n", cutGetTimerValue(timer_CPU));

    Cleanup();
}

void Cleanup(void)
{
    // Free device memory
    if (d_MatA)
        cudaFree(d_MatA);
    if (d_VecV)
        cudaFree(d_VecV);
    if (d_VecW)
        cudaFree(d_VecW);
    if (d_NormW)
        cudaFree(d_NormW);
		
    // Free host memory
    if (h_MatA)
        free(h_MatA);
    if (h_VecV)
        free(h_VecV);
    if (h_VecW)
        free(h_VecW);
    if (h_NormW)
        free(h_NormW);
    
    exit(0);
}

// Allocates an array with zero value.
void InitOne(float* data, int n)
{
    for (int i = 0; i < n; i++)
        data[i] = 0;
	data[0]=1;
}

void UploadArray(float* data, int n)
{
    int total = n*n;
    for (int i = 0; i < total; i++)
    {
        data[i] = (int) (rand() % (int)(101));
    }
}

// Obtain program arguments
void ParseArguments(int argc, char** argv)
{
    for (int i = 0; i < argc; ++i) 
    {
        if (strcmp(argv[i], "--size") == 0 || strcmp(argv[i], "-size") == 0)
        {
            GlobalSize = atoi(argv[i+1]);
		    i = i + 1;
        }
        if (strcmp(argv[i], "--max_iteration") == 0 || strcmp(argv[i], "-max_iteration") == 0)
        {
            max_iteration = atoi(argv[i+1]);
		    i = i + 1;
        }
        if (strcmp(argv[i], "--seed") == 0 || strcmp(argv[i], "-seed") == 0)
        {
            int seed = atoi(argv[i+1]);
            srand(seed);
        }
    }
}


void checkCardVersion()
{
   cudaDeviceProp prop;
   
   cudaGetDeviceProperties(&prop, 0);
   
   printf("This GPU has major architecture %d, minor %d \n",prop.major,prop.minor);
   if(prop.major < 2)
   {
      fprintf(stderr,"Need compute capability 2 or higher.\n");
      exit(1);
   }
}



/*****************************************************************************
This function finds the product of Matrix A and vector V
*****************************************************************************/

// ****************************************************************************************************************************************************/
// parallelization method for the Matrix-vector multiplication as follows: 

// each thread handle a multiplication of each row of Matrix A and vector V;

// The share memory is limited for a block, instead of reading an entire row of matrix A or vector V from global memory to share memory, 
// a square submatrix of A is shared by a block, the size of square submatrix is BLOCK_SIZE*BLOCK_SIZE; Thus, a for-loop is used to
// handle a multiplication of each row of Matrix A and vector V step by step. In eacg step, two subvectors with size BLOCK_SIZE is multiplied.
//*****************************************************************************************************************************************************/

__global__ void Av_Product(float* g_MatA, float* g_VecV, float* g_VecW, int N)
{
    // Block index
    int bx = blockIdx.x;

    // Thread index
    int tx = threadIdx.x;

    float Csub = 0;
    int row = tx + bx * BLOCK_SIZE;
    int row_offset = N * row;

    
    if(row < N) {
        for (int j = 0; j < N; j++) {
            Csub += g_MatA[j + row_offset] * g_VecV[j];
        }
    }

    __syncthreads();

    if(row < N) {
        g_VecW[ row ] = Csub;
    }
}





__global__ void ComputeLamda( float* g_VecV, float* g_VecW, float * g_Lamda,int N)
{
  // shared memory size declared at kernel launch
  unsigned int tid = threadIdx.x;
  unsigned int offset = blockIdx.x * blockDim.x;
  unsigned int globalid = offset + tid;

  // For thread ids greater than data space
  if (globalid < N) {
     g_VecV[globalid] =  g_VecV[globalid] * g_VecW[globalid];
  }

  // each thread loads one element from global to shared mem
  __syncthreads();

  // do reduction in shared mem
  for (unsigned int s=blockDim.x / 2; s > 0; s = s >> 1) {
     if ((tid < s) && (globalid + s < N)) {
         g_VecV[globalid] = g_VecV[globalid] + g_VecV[globalid + s];
     }
     __syncthreads();
  }
   // atomic operations:
  if (tid == 0) atomicAdd(g_Lamda,g_VecV[offset]);
}



/****************************************************
Normalizes vector W : W/norm(W)
****************************************************/
__global__ void FindNormW(float* g_VecW, float* g_tempV, float * g_NormW, int N)
{
  // shared memory size declared at kernel launch
  unsigned int tid = threadIdx.x;
  unsigned int offset = blockIdx.x*blockDim.x;
  unsigned int globalid = blockIdx.x*blockDim.x + threadIdx.x;

  // For thread ids greater than data space
  if (globalid < N) {
     g_tempV[globalid] =  g_VecW[globalid] * g_VecW[globalid];
  }

  __syncthreads();

  // do reduction in shared mem
  for (unsigned int s=blockDim.x / 2; s > 0; s = s >> 1) {
     if ((tid < s) && (globalid + s < N)) {
         g_tempV[globalid] = g_tempV[globalid] + g_tempV[globalid + s];
     }
     __syncthreads();
  }
   // atomic operations:
  if (tid == 0) atomicAdd(g_NormW,g_tempV[offset]);
}


__global__ void NormalizeW(float* g_VecW, float * g_NormW, float* g_VecV, int N)
{
  unsigned int globalid = blockIdx.x*blockDim.x + threadIdx.x;
  __syncthreads();

  // For thread ids greater than data space
  if (globalid < N) {
     g_VecV[globalid] = g_VecW[globalid] / g_NormW[0];
  }
}
