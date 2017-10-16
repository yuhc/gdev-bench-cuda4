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

struct timeval tv;
struct timeval tv_total_start, tv_total_end;
float total;
struct timeval tv_h2d_start, tv_h2d_end;
float h2d;
struct timeval tv_d2h_start, tv_d2h_end;
float d2h;
struct timeval tv_exec_start, tv_exec_end;
struct timeval tv_mem_alloc_start;
struct timeval tv_close_start;
float mem_alloc;
float exec;
float init_gpu;
float close_gpu;

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

CUresult pathfinder_launch(CUmodule mod, int gdx, int bdx, int iteration,
                CUdeviceptr gpuWall, CUdeviceptr gpuSrc, CUdeviceptr gpuResults,
                int cols, int rows, int startStep, int border)
{
    void* param[] = {&iteration, &gpuWall, &gpuSrc, &gpuResults, &cols, &rows,
        &startStep, &border};
    CUfunction f;
    CUresult res;

    res = cuModuleGetFunction(&f, mod, "_Z14dynproc_kerneliPiS_S_iiii");
    if (res != CUDA_SUCCESS) {
        printf("cuModuleGetFunction failed: res = %u\n", res);
        return res;
    }

    /* shared memory size is known in the kernel image. */
    res = cuLaunchKernel(f, gdx, 1, 1, bdx, 1, 1, 0, 0, (void**) param, NULL);
    if (res != CUDA_SUCCESS) {
        printf("cuLaunchKernel(euclid) failed: res = %u\n", res);
        return res;
    }

    return CUDA_SUCCESS;
}

/*
   compute N time steps
*/
int calc_path(CUmodule mod, CUdeviceptr gpuWall, CUdeviceptr gpuResult[2], int rows, int cols, \
     int pyramid_height, int blockCols, int borderCols)
{
    int src = 1, dst = 0;
    int t;
    for (t = 0; t < rows-1; t+=pyramid_height) {
        int temp = src;
        src = dst;
        dst = temp;
        pathfinder_launch(mod, blockCols, BLOCK_SIZE,
                MIN(pyramid_height, rows-t-1), gpuWall, gpuResult[src],
                gpuResult[dst], cols,rows, t, borderCols);
    }
    return dst;
}

int run(int argc, char** argv)
{
    init(argc, argv);

    CUcontext ctx;
    CUmodule mod;
    CUresult res;

    /* --------------- pyramid parameters --------------- */
    int borderCols = (pyramid_height)*HALO;
    int smallBlockCol = BLOCK_SIZE-(pyramid_height)*HALO*2;
    int blockCols = cols/smallBlockCol+((cols%smallBlockCol==0)?0:1);

    printf("pyramidHeight: %d\ngridSize: [%d]\nborder:[%d]\nblockSize: %d\nblockGrid:[%d]\ntargetBlock:[%d]\n",\
    pyramid_height, cols, borderCols, BLOCK_SIZE, blockCols, smallBlockCol);

    CUdeviceptr gpuWall, gpuResult[2];
    int size = rows*cols;

    /* call our common CUDA initialization utility function. */
	gettimeofday(&tv_total_start, NULL);
    res = cuda_driver_api_init(&ctx, &mod, "./pathfinder.cubin");
    if (res != CUDA_SUCCESS) {
        printf("cuda_driver_api_init failed: res = %u\n", res);
        return -1;
    }

    gettimeofday(&tv_mem_alloc_start, NULL);
	tvsub(&tv_mem_alloc_start, &tv_total_start, &tv);
	init_gpu = tv.tv_sec * 1000.0 + (float) tv.tv_usec / 1000.0;

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

    gettimeofday(&tv_h2d_start, NULL);
    tvsub(&tv_h2d_start, &tv_mem_alloc_start, &tv);
	mem_alloc = tv.tv_sec * 1000.0 + (float) tv.tv_usec / 1000.0;

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

    gettimeofday(&tv_h2d_end, NULL);
    tvsub(&tv_h2d_end, &tv_h2d_start, &tv);
    h2d = tv.tv_sec * 1000.0 + (float) tv.tv_usec / 1000.0;

    int final_ret;

int tc;
for (tc = 0; tc < 100; tc++) {

    gettimeofday(&tv_h2d_end, NULL);
    final_ret = calc_path(mod, gpuWall, gpuResult, rows, cols, pyramid_height, blockCols, borderCols);

    gettimeofday(&tv_exec_end, NULL);
    tvsub(&tv_exec_end, &tv_h2d_end, &tv);
    exec += tv.tv_sec * 1000.0 + (float) tv.tv_usec / 1000.0;

}

    /* Copy data from device memory to main memory */
    res = cuMemcpyDtoH(result, gpuResult[final_ret], sizeof(float) * cols);
    if (res != CUDA_SUCCESS) {
        printf("cuMemcpyHtoD failed: res = %u\n", res);
        return -1;
    }

	gettimeofday(&tv_d2h_end, NULL);
    tvsub(&tv_d2h_end, &tv_exec_end, &tv);
	d2h = tv.tv_sec * 1000.0 + (float) tv.tv_usec / 1000.0;

    cuMemFree(gpuWall);
    cuMemFree(gpuResult[0]);
    cuMemFree(gpuResult[1]);

    res = cuda_driver_api_exit(ctx, mod);
    if (res != CUDA_SUCCESS) {
        printf("cuda_driver_api_exit failed: res = %u\n", res);
        return -1;
    }

	gettimeofday(&tv_total_end, NULL);
	tvsub(&tv_total_end, &tv_d2h_end, &tv);
	close_gpu = tv.tv_sec * 1000.0 + (float) tv.tv_usec / 1000.0;

	tvsub(&tv_total_end, &tv_total_start, &tv);
	total = tv.tv_sec * 1000.0 + (float) tv.tv_usec / 1000.0;

	printf("Init: %f\n", init_gpu);
	printf("MemAlloc: %f\n", mem_alloc);
	printf("HtoD: %f\n", h2d);
	printf("Exec: %f\n", exec);
	printf("DtoH: %f\n", d2h);
	printf("Close: %f\n", close_gpu);
	printf("Total: %f\n", total);

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

    return 0;
}
