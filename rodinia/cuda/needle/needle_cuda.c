#define LIMIT -999
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda.h>
#include <sys/time.h>

#include "needle_cuda.h"
#include "util.h"

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
int runTest( int argc, char** argv);


int blosum62[24][24] = {
{ 4, -1, -2, -2,  0, -1, -1,  0, -2, -1, -1, -1, -1, -2, -1,  1,  0, -3, -2,  0, -2, -1,  0, -4},
{-1,  5,  0, -2, -3,  1,  0, -2,  0, -3, -2,  2, -1, -3, -2, -1, -1, -3, -2, -3, -1,  0, -1, -4},
{-2,  0,  6,  1, -3,  0,  0,  0,  1, -3, -3,  0, -2, -3, -2,  1,  0, -4, -2, -3,  3,  0, -1, -4},
{-2, -2,  1,  6, -3,  0,  2, -1, -1, -3, -4, -1, -3, -3, -1,  0, -1, -4, -3, -3,  4,  1, -1, -4},
{ 0, -3, -3, -3,  9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1, -3, -3, -2, -4},
{-1,  1,  0,  0, -3,  5,  2, -2,  0, -3, -2,  1,  0, -3, -1,  0, -1, -2, -1, -2,  0,  3, -1, -4},
{-1,  0,  0,  2, -4,  2,  5, -2,  0, -3, -3,  1, -2, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1, -4},
{ 0, -2,  0, -1, -3, -2, -2,  6, -2, -4, -4, -2, -3, -3, -2,  0, -2, -2, -3, -3, -1, -2, -1, -4},
{-2,  0,  1, -1, -3,  0,  0, -2,  8, -3, -3, -1, -2, -1, -2, -1, -2, -2,  2, -3,  0,  0, -1, -4},
{-1, -3, -3, -3, -1, -3, -3, -4, -3,  4,  2, -3,  1,  0, -3, -2, -1, -3, -1,  3, -3, -3, -1, -4},
{-1, -2, -3, -4, -1, -2, -3, -4, -3,  2,  4, -2,  2,  0, -3, -2, -1, -2, -1,  1, -4, -3, -1, -4},
{-1,  2,  0, -1, -3,  1,  1, -2, -1, -3, -2,  5, -1, -3, -1,  0, -1, -3, -2, -2,  0,  1, -1, -4},
{-1, -1, -2, -3, -1,  0, -2, -3, -2,  1,  2, -1,  5,  0, -2, -1, -1, -1, -1,  1, -3, -1, -1, -4},
{-2, -3, -3, -3, -2, -3, -3, -3, -1,  0,  0, -3,  0,  6, -4, -2, -2,  1,  3, -1, -3, -3, -1, -4},
{-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4,  7, -1, -1, -4, -3, -2, -2, -1, -2, -4},
{ 1, -1,  1,  0, -1,  0,  0,  0, -1, -2, -2,  0, -1, -2, -1,  4,  1, -3, -2, -2,  0,  0,  0, -4},
{ 0, -1,  0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1,  1,  5, -2, -2,  0, -1, -1,  0, -4},
{-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1,  1, -4, -3, -2, 11,  2, -3, -4, -3, -2, -4},
{-2, -2, -2, -3, -2, -1, -2, -3,  2, -1, -1, -2, -1,  3, -3, -2, -2,  2,  7, -1, -3, -2, -1, -4},
{ 0, -3, -3, -3, -1, -2, -2, -3, -3,  3,  1, -2,  1, -1, -2, -2,  0, -3, -1,  4, -3, -2, -1, -4},
{-2, -1,  3,  4, -3,  0,  1, -1,  0, -3, -4,  0, -3, -3, -2,  0, -1, -4, -3, -3,  4,  1, -1, -4},
{-1,  0,  0,  1, -3,  3,  4, -2,  0, -3, -3,  1, -1, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1, -4},
{ 0, -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2,  0,  0, -2, -1, -1, -1, -1, -1, -4},
{-4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4,  1}
};

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

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main( int argc, char** argv) 
{
    int rt = runTest( argc, argv);
    if (rt < 0) return rt;

    return EXIT_SUCCESS;
}

void usage(int argc, char **argv)
{
	fprintf(stderr, "Usage: %s <max_rows/max_cols> <penalty> \n", argv[0]);
	fprintf(stderr, "\t<dimension>  - x and y dimensions\n");
	fprintf(stderr, "\t<penalty> - penalty(positive integer)\n");
	exit(1);
}

CUresult needle_launch(CUmodule mod, int gdx, int gdy, int bdx, int bdy,
        CUdeviceptr referrence_cuda, CUdeviceptr matrix_cuda, CUdeviceptr matrix_cuda_out,
        int max_cols, int penalty, int i, int block_width)
{
    void* param[] = {&referrence_cuda, &matrix_cuda, &matrix_cuda_out, &max_cols, &penalty, &i, &block_width};
    CUfunction f;
    CUresult res;

    res = cuModuleGetFunction(&f, mod, "_Z20needle_cuda_shared_1PiS_S_iiii");
    if (res != CUDA_SUCCESS) {
        printf("cuModuleGetFunction failed: res = %u\n", res);
        return res;
    }

    /* shared memory size is known in the kernel image. */
    res = cuLaunchKernel(f, gdx, gdy, 1, bdx, bdy, 1, 0, 0, (void**) param, NULL);
    if (res != CUDA_SUCCESS) {
        printf("cuLaunchKernel(euclid) failed: res = %u\n", res);
        return res;
    }

    return CUDA_SUCCESS;
}

CUresult needle_launch2(CUmodule mod, int gdx, int gdy, int bdx, int bdy,
        CUdeviceptr referrence_cuda, CUdeviceptr matrix_cuda, CUdeviceptr matrix_cuda_out,
        int max_cols, int penalty, int i, int block_width)
{
    void* param[] = {&referrence_cuda, &matrix_cuda, &matrix_cuda_out, &max_cols, &penalty, &i, &block_width};
    CUfunction f;
    CUresult res;

    res = cuModuleGetFunction(&f, mod, "_Z20needle_cuda_shared_2PiS_S_iiii");
    if (res != CUDA_SUCCESS) {
        printf("cuModuleGetFunction failed: res = %u\n", res);
        return res;
    }

    /* shared memory size is known in the kernel image. */
    res = cuLaunchKernel(f, gdx, gdy, 1, bdx, bdy, 1, 0, 0, (void**) param, NULL);
    if (res != CUDA_SUCCESS) {
        printf("cuLaunchKernel(euclid) failed: res = %u\n", res);
        return res;
    }

    return CUDA_SUCCESS;
}

int runTest( int argc, char** argv) 
{
    int max_rows, max_cols, penalty;
    int *input_itemsets, *output_itemsets, *referrence;
	int size;

    // the lengths of the two sequences should be able to divided by 16.
	// And at current stage  max_rows needs to equal max_cols
	if (argc == 3)
	{
		max_rows = atoi(argv[1]);
		max_cols = atoi(argv[1]);
		penalty = atoi(argv[2]);
	}
    else{
	usage(argc, argv);
    }

	if(atoi(argv[1])%16!=0){
	fprintf(stderr,"The dimension values must be a multiple of 16\n");
	exit(1);
	}

	max_rows = max_rows + 1;
	max_cols = max_cols + 1;
	referrence = (int *)malloc( max_rows * max_cols * sizeof(int) );
    input_itemsets = (int *)malloc( max_rows * max_cols * sizeof(int) );
	output_itemsets = (int *)malloc( max_rows * max_cols * sizeof(int) );

	if (!input_itemsets)
		fprintf(stderr, "error: can not allocate memory");

    srand ( 7 );

    int i, j;
    for (i = 0 ; i < max_cols; i++){
		for (j = 0 ; j < max_rows; j++){
			input_itemsets[i*max_cols+j] = 0;
		}
	}

	printf("Start Needleman-Wunsch\n");

	for(i=1; i< max_rows ; i++){    //please define your own sequence.
       input_itemsets[i*max_cols] = rand() % 10 + 1;
	}
    for(j=1; j< max_cols ; j++){    //please define your own sequence.
       input_itemsets[j] = rand() % 10 + 1;
	}

	for (i = 1 ; i < max_cols; i++){
		for (j = 1 ; j < max_rows; j++){
		referrence[i*max_cols+j] = blosum62[input_itemsets[i*max_cols]][input_itemsets[j]];
		}
	}

    for(i = 1; i< max_rows ; i++)
       input_itemsets[i*max_cols] = -i * penalty;
	for(j = 1; j< max_cols ; j++)
       input_itemsets[j] = -j * penalty;

    /* call our common CUDA initialization utility function. */
    CUcontext ctx;
    CUmodule mod;
    CUresult res;
    CUdeviceptr referrence_cuda, matrix_cuda, matrix_cuda_out;

	gettimeofday(&tv_total_start, NULL);
    res = cuda_driver_api_init(&ctx, &mod, "./needle.cubin");
    if (res != CUDA_SUCCESS) {
        printf("cuda_driver_api_init failed: res = %u\n", res);
        return -1;
    }

    size = max_cols * max_rows;

    gettimeofday(&tv_mem_alloc_start, NULL);
	tvsub(&tv_mem_alloc_start, &tv_total_start, &tv);
	init_gpu = tv.tv_sec * 1000.0 + (float) tv.tv_usec / 1000.0;

    /* Allocate device memory */
    res = cuMemAlloc(&referrence_cuda, sizeof(int) * size);
    if (res != CUDA_SUCCESS) {
        printf("cuMemAlloc failed: res = %u\n", res);
        return -1;
    }

    res = cuMemAlloc(&matrix_cuda, sizeof(int) * size);
    if (res != CUDA_SUCCESS) {
        printf("cuMemAlloc failed: res = %u\n", res);
        return -1;
    }

    res = cuMemAlloc(&matrix_cuda_out, sizeof(int) * size);
    if (res != CUDA_SUCCESS) {
        printf("cuMemAlloc failed: res = %u\n", res);
        return -1;
    }

    gettimeofday(&tv_h2d_start, NULL);
    tvsub(&tv_h2d_start, &tv_mem_alloc_start, &tv);
	mem_alloc = tv.tv_sec * 1000.0 + (float) tv.tv_usec / 1000.0;

    /* Copy data from main memory to device memory */
    res = cuMemcpyHtoD(referrence_cuda, referrence, sizeof(int) * size);
    if (res != CUDA_SUCCESS) {
        printf("cuMemcpyHtoD failed: res = %u\n", res);
        return -1;
    }

    res = cuMemcpyHtoD(matrix_cuda, input_itemsets, sizeof(int) * size);
    if (res != CUDA_SUCCESS) {
        printf("cuMemcpyHtoD failed: res = %u\n", res);
        return -1;
    }

	int block_width = ( max_cols - 1 )/BLOCK_SIZE;

    gettimeofday(&tv_h2d_end, NULL);
    tvsub(&tv_h2d_end, &tv_h2d_start, &tv);
    h2d = tv.tv_sec * 1000.0 + (float) tv.tv_usec / 1000.0;

	printf("Processing top-left matrix\n");

	//process top-left matrix
	for(i = 1 ; i <= block_width ; i++){
        needle_launch(mod, i, 1, BLOCK_SIZE, 1, referrence_cuda, matrix_cuda,
                matrix_cuda_out, max_cols, penalty, i, block_width);
	}
    	cuCtxSynchronize();

	printf("Processing bottom-right matrix\n");
    //process bottom-right matrix
	for(i = block_width - 1  ; i >= 1 ; i--){
        needle_launch2(mod, i, 1, BLOCK_SIZE, 1, referrence_cuda, matrix_cuda,
                matrix_cuda_out, max_cols, penalty, i, block_width);
	}

    cuCtxSynchronize();
    gettimeofday(&tv_exec_end, NULL);
    tvsub(&tv_exec_end, &tv_h2d_end, &tv);
    exec = tv.tv_sec * 1000.0 + (float) tv.tv_usec / 1000.0;

    /* Copy data from device memory to main memory */
    res = cuMemcpyDtoH(output_itemsets, matrix_cuda, sizeof(int) * size);
    if (res != CUDA_SUCCESS) {
        printf("cuMemcpyHtoD failed: res = %u\n", res);
        return -1;
    }

	gettimeofday(&tv_d2h_end, NULL);
    tvsub(&tv_d2h_end, &tv_exec_end, &tv);
	d2h = tv.tv_sec * 1000.0 + (float) tv.tv_usec / 1000.0;

	cuMemFree(referrence_cuda);
	cuMemFree(matrix_cuda);
	cuMemFree(matrix_cuda_out);

	res = cuda_driver_api_exit(ctx, mod);
	if (res != CUDA_SUCCESS) {
		printf("cuda_driver_api_exit faild: res = %u\n", res);
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

//#define TRACEBACK
#ifdef TRACEBACK
	FILE *fpo = fopen("result.txt","w");
	fprintf(fpo, "print traceback value GPU:\n");

	for (i = max_rows - 2,  j = max_rows - 2; i>=0, j>=0;){
		int nw, n, w, traceback;
		if ( i == max_rows - 2 && j == max_rows - 2 )
			fprintf(fpo, "%d ", output_itemsets[ i * max_cols + j]); //print the first element
		if ( i == 0 && j == 0 )
           break;
		if ( i > 0 && j > 0 ){
			nw = output_itemsets[(i - 1) * max_cols + j - 1];
		    w  = output_itemsets[ i * max_cols + j - 1 ];
            n  = output_itemsets[(i - 1) * max_cols + j];
		}
		else if ( i == 0 ){
		    nw = n = LIMIT;
		    w  = output_itemsets[ i * max_cols + j - 1 ];
		}
		else if ( j == 0 ){
		    nw = w = LIMIT;
            n  = output_itemsets[(i - 1) * max_cols + j];
		}
		else{
		}

		//traceback = maximum(nw, w, n);
		int new_nw, new_w, new_n;
		new_nw = nw + referrence[i * max_cols + j];
		new_w = w - penalty;
		new_n = n - penalty;

		traceback = maximum(new_nw, new_w, new_n);
		if(traceback == new_nw)
			traceback = nw;
		if(traceback == new_w)
			traceback = w;
		if(traceback == new_n)
            traceback = n;

		fprintf(fpo, "%d ", traceback);

		if(traceback == nw )
		{i--; j--; continue;}

        else if(traceback == w )
		{j--; continue;}

        else if(traceback == n )
		{i--; continue;}

		else
		;
	}

	fclose(fpo);
#endif

	free(referrence);
	free(input_itemsets);
	free(output_itemsets);

    return 0;
}

