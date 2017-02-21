#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include <cuda.h>
#include "util.h"

#include "pathfinder_cuda.h"

#define BLOCK_SIZE 256
#define STR_SIZE 256
#define DEVICE 0

//#define BENCH_PRINT

void init(int argc, char** argv);
int run(int argc, char** argv);

int rows, cols;
int* data;
int** wall;
int* result;
#define M_SEED 9
int pyramid_height;

int main(int argc, char** argv)
{
    int rt;
    rt = run(argc,argv);
    if (rt < 0) return rt;

    return EXIT_SUCCESS;
}

void
init(int argc, char** argv)
{
    if(argc==4){
        cols = atoi(argv[1]);
        rows = atoi(argv[2]);
                pyramid_height=atoi(argv[3]);
    }else{
                printf("Usage: dynproc row_len col_len pyramid_height\n");
                exit(0);
        }
    data = (int*)malloc(sizeof(int) * rows * cols);
    wall = (int**)malloc(sizeof(int*) * rows);
    int n;
    for(n=0; n<rows; n++)
        wall[n]=data+cols*n;
    result = (int*)malloc(sizeof(int) * cols);

    int seed = M_SEED;
    srand(seed);

    int i, j;
    for (i = 0; i < rows; i++)
    {
        for (j = 0; j < cols; j++)
        {
            wall[i][j] = rand() % 10;
        }
    }
#ifdef BENCH_PRINT
    for (i = 0; i < rows; i++)
    {
        for (j = 0; j < cols; j++)
        {
            printf("%d ",wall[i][j]) ;
        }
        printf("\n") ;
    }
#endif
}

void
fatal(char *s)
{
    fprintf(stderr, "error: %s\n", s);
}

/*
   compute N time steps
*/
/*
int calc_path(int *gpuWall, int *gpuResult[2], int rows, int cols, \
     int pyramid_height, int blockCols, int borderCols)
{
        dim3 dimBlock(BLOCK_SIZE);
        dim3 dimGrid(blockCols);

        int src = 1, dst = 0;
    for (int t = 0; t < rows-1; t+=pyramid_height) {
            int temp = src;
            src = dst;
            dst = temp;
            dynproc_kernel<<<dimGrid, dimBlock>>>(
                MIN(pyramid_height, rows-t-1), 
                gpuWall, gpuResult[src], gpuResult[dst],
                cols,rows, t, borderCols);
    }
        return dst;
}
*/

int run(int argc, char** argv)
{
    init(argc, argv);

    CUcontext ctx;
    CUmodule mod;
    CUresult res;

    /* call our common CUDA initialization utility function. */
    res = cuda_driver_api_init(&ctx, &mod, "./pathfinder.cubin");
    if (res != CUDA_SUCCESS) {
        printf("cuda_driver_api_init failed: res = %u\n", res);
        return -1;
    }

    /* --------------- pyramid parameters --------------- */
    int borderCols = (pyramid_height)*HALO;
    int smallBlockCol = BLOCK_SIZE-(pyramid_height)*HALO*2;
    int blockCols = cols/smallBlockCol+((cols%smallBlockCol==0)?0:1);

    printf("pyramidHeight: %d\ngridSize: [%d]\nborder:[%d]\nblockSize: %d\nblockGrid:[%d]\ntargetBlock:[%d]\n",\
    pyramid_height, cols, borderCols, BLOCK_SIZE, blockCols, smallBlockCol);

    CUdeviceptr gpuWall, gpuResult[2];
    int size = rows*cols;

    res = cuMemAlloc(&gpuResult[0], sizeof(int) * cols);
    if (res != CUDA_SUCCESS) {
        printf("cuMemAlloc failed: res = %u\n", res);
        return -1;
    }

    res = cuMemAlloc(&gpuResult[1], sizeof(int) * cols);
    if (res != CUDA_SUCCESS) {
        printf("cuMemAlloc failed: res = %u\n", res);
        return -1;
    }
    res = cuMemAlloc(&gpuWall, sizeof(int) * (size - cols));
    if (res != CUDA_SUCCESS) {
        printf("cuMemAlloc failed: res = %u\n", res);
        return -1;
    }

    res = cuMemcpyHtoD(gpuResult[0], data, sizeof(int) * cols);
    if (res != CUDA_SUCCESS) {
        printf("cuMemcpyHtoD failed: res = %u\n", res);
        return -1;
    }

    res = cuMemcpyHtoD(gpuWall, data+cols, sizeof(int) * (size - cols));
    if (res != CUDA_SUCCESS) {
        printf("cuMemcpyHtoD failed: res = %u\n", res);
        return -1;
    }

    // begin timing kernels
    struct timeval time_start;
    gettimeofday(&time_start, NULL);

    int final_ret;
    //final_ret = calc_path(gpuWall, gpuResult, rows, cols, pyramid_height, blockCols, borderCols);

    /* Copy data from device memory to main memory */
    res = cuMemcpyDtoH(result, gpuResult[final_ret], sizeof(float) * cols);
    if (res != CUDA_SUCCESS) {
        printf("cuMemcpyHtoD failed: res = %u\n", res);
        return -1;
    }

    // end timing kernels
    unsigned int totalKernelTime = 0;
    struct timeval time_end;
    gettimeofday(&time_end, NULL);
    totalKernelTime = (time_end.tv_sec * 1000000 + time_end.tv_usec) - (time_start.tv_sec * 1000000 + time_start.tv_usec);
    printf("Time for CUDA kernels:\t%f sec\n",totalKernelTime * 1e-6);

	cuMemFree(gpuWall);
	cuMemFree(gpuResult[0]);
	cuMemFree(gpuResult[1]);

#ifdef BENCH_PRINT
    for (int i = 0; i < cols; i++)
            printf("%d ",data[i]) ;
    printf("\n") ;
    for (int i = 0; i < cols; i++)
            printf("%d ",result[i]) ;
    printf("\n") ;
#endif

    free(data);
    free(wall);
    free(result);

    res = cuda_driver_api_exit(ctx, mod);
    if (res != CUDA_SUCCESS) {
        printf("cuda_driver_api_exit failed: res = %u\n", res);
        return -1;
    }

    return 0;
}
