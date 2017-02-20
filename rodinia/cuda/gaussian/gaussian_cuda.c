/*-----------------------------------------------------------
 ** gaussian.cu -- The program is to solve a linear system Ax = b
 **   by using Gaussian Elimination. The algorithm on page 101
 **   ("Foundations of Parallel Programming") is used.  
 **   The sequential version is gaussian.c.  This parallel 
 **   implementation converts three independent for() loops 
 **   into three Fans.  Use the data file ge_3.dat to verify 
 **   the correction of the output. 
 **
 ** Written by Andreas Kura, 02/15/95
 ** Modified by Chong-wei Xu, 04/20/95
 ** Modified by Chris Gregg for CUDA, 07/20/2009
 ** Modified by Hangchen Yu for GDEV, 02/19/2017
 **-----------------------------------------------------------
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>

#include <cuda.h>

#include "util.h"

#define MAXBLOCKSIZE 512

int Size;
float *a, *b, *finalVec;
float *m;

FILE *fp;

void usage(int argc, char *argv[]);
void InitProblemOnce(char *filename);
void InitPerRun();
int ForwardSub(CUmodule mod);
void BackSub();
void InitMat(float *ary, int nrow, int ncol);
void InitAry(float *ary, int ary_size);
void PrintMat(float *ary, int nrow, int ncolumn);
void PrintAry(float *ary, int ary_size);
void PrintDeviceProperties();

unsigned int totalKernelTime = 0;
unsigned int total_time = 0;

//=========================================================================
// KERNEL CODE
//=========================================================================

CUresult gaussian_launch(CUmodule mod, int gdx, int gdy, int bdx, int bdy, CUdeviceptr m_cuda,
        CUdeviceptr a_cuda, int Size, int t)
{
    void* param[] = {&m_cuda, &a_cuda, &Size, &t};
    CUfunction f;
    CUresult res;

    res = cuModuleGetFunction(&f, mod, "_Z4Fan1PfS_ii");
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

CUresult gaussian_launch2(CUmodule mod, int gdx, int gdy, int bdx, int bdy, CUdeviceptr m_cuda,
        CUdeviceptr a_cuda, CUdeviceptr b_cuda, int Size, int j1, int t)
{
    void* param[] = {&m_cuda, &a_cuda, &b_cuda, &Size, &j1, &t};
    CUfunction f;
    CUresult res;

    res = cuModuleGetFunction(&f, mod, "_Z4Fan2PfS_S_iii");
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

int main(int argc, char *argv []){
    CUcontext ctx;
    CUmodule mod;
    CUresult res;
    int rt;

    int verbose = 1;

    usage(argc, argv);
    InitProblemOnce(argv[1]);
    if (argc > 2) {
        if (!strcmp(argv[2],"-q")) verbose = 0;
    }
    InitPerRun();

    if (verbose) {
        printf("Matrix m is: \n");
        PrintMat(m, Size, Size);

        printf("Matrix a is: \n");
        PrintMat(a, Size, Size);

        printf("Array b is: \n");
        PrintAry(b, Size);
    }

    /* call our common CUDA initialization utility function. */
    res = cuda_driver_api_init(&ctx, &mod, "./gaussian.cubin");
    if (res != CUDA_SUCCESS) {
        printf("cuda_driver_api_init failed: res = %u\n", res);
        return -1;
    }

    // begin timing kernels
    struct timeval tot_time_start;
    gettimeofday(&tot_time_start, NULL);

    rt = ForwardSub(mod);
    if (rt < 0) return -1;
    BackSub();
    if (verbose) {
        printf("The final solution is: \n");
        PrintAry(finalVec,Size);
    }

    // end timing kernels
    struct timeval tot_time_end;
    gettimeofday(&tot_time_end, NULL);
    total_time = (tot_time_end.tv_sec * 1000000 + tot_time_end.tv_usec) - (tot_time_start.tv_sec * 1000000 + tot_time_start.tv_usec);

    printf("\nTime total (including memory transfers)\t%f sec\n", total_time * 1e-6);
    printf("Time for CUDA kernels:\t%f sec\n",totalKernelTime * 1e-6);

    free(m);
    free(a);
    free(b);

    res = cuda_driver_api_exit(ctx, mod);
    if (res != CUDA_SUCCESS) {
        printf("cuda_driver_api_exit failed: res = %u\n", res);
        return -1;
    }

    return 0;
}

void usage(int argc, char *argv[])
{
    if (argc < 2) {
        printf("Usage: gaussian matrix.txt [-q]\n\n");
        printf("-q (quiet) suppresses printing the matrix and result values.\n");
        printf("The first line of the file contains the dimension of the matrix, n.");
        printf("The second line of the file is a newline.\n");
        printf("The next n lines contain n tab separated values for the matrix.");
        printf("The next line of the file is a newline.\n");
        printf("The next line of the file is a 1xn vector with tab separated values.\n");
        printf("The next line of the file is a newline. (optional)\n");
        printf("The final line of the file is the pre-computed solution. (optional)\n");
        printf("Example: matrix4.txt:\n");
        printf("4\n");
        printf("\n");
        printf("-0.6    -0.5    0.7 0.3\n");
        printf("-0.3    -0.9    0.3 0.7\n");
        printf("-0.4    -0.5    -0.3    -0.8\n");   
        printf("0.0 -0.1    0.2 0.9\n");
        printf("\n");
        printf("-0.85   -0.68   0.24    -0.53\n");  
        printf("\n");
        printf("0.7 0.0 -0.4    -0.5\n");
        exit(0);
    }
}

/*------------------------------------------------------
 ** PrintMat() -- Print the contents of the matrix
 **------------------------------------------------------
 */
void PrintMat(float *ary, int nrow, int ncol)
{
    int i, j;

    for (i=0; i<nrow; i++) {
        for (j=0; j<ncol; j++) {
            printf("%8.2f ", *(ary+Size*i+j));
        }
        printf("\n");
    }
    printf("\n");
}

/*------------------------------------------------------
 ** PrintAry() -- Print the contents of the array (vector)
 **------------------------------------------------------
 */
void PrintAry(float *ary, int ary_size)
{
    int i;
    for (i=0; i<ary_size; i++) {
        printf("%.2f ", ary[i]);
    }
    printf("\n\n");
}

/*------------------------------------------------------
 ** InitProblemOnce -- Initialize all of matrices and
 ** vectors by opening a data file specified by the user.
 **
 ** We used dynamic array *a, *b, and *m to allocate
 ** the memory storages.
 **------------------------------------------------------
 */
void InitProblemOnce(char *filename)
{
    //char *filename = argv[1];

    //printf("Enter the data file name: ");
    //scanf("%s", filename);
    //printf("The file name is: %s\n", filename);

    fp = fopen(filename, "r");

    fscanf(fp, "%d", &Size);

    a = (float *) malloc(Size * Size * sizeof(float));

    InitMat(a, Size, Size);
    //printf("The input matrix a is:\n");
    //PrintMat(a, Size, Size);
    b = (float *) malloc(Size * sizeof(float));

    InitAry(b, Size);
    //printf("The input array b is:\n");
    //PrintAry(b, Size);

    m = (float *) malloc(Size * Size * sizeof(float));
}

void InitMat(float *ary, int nrow, int ncol)
{
    int i, j;

    for (i=0; i<nrow; i++) {
        for (j=0; j<ncol; j++) {
            fscanf(fp, "%f",  ary+Size*i+j);
        }
    }
}

void InitAry(float *ary, int ary_size)
{
    int i;

    for (i=0; i<ary_size; i++) {
        fscanf(fp, "%f",  &ary[i]);
    }
}

/*------------------------------------------------------
 ** InitPerRun() -- Initialize the contents of the
 ** multipier matrix **m
 **------------------------------------------------------
 */
void InitPerRun() 
{
    int i;
    for (i=0; i<Size*Size; i++)
            *(m+i) = 0.0;
}

/*------------------------------------------------------
 ** ForwardSub() -- Forward substitution of Gaussian
 ** elimination.
 **------------------------------------------------------
 */
int ForwardSub(CUmodule mod)
{
    int t;
    CUdeviceptr m_cuda, a_cuda, b_cuda;
    CUresult res;

    /* Allocate device memory */
    res = cuMemAlloc(&m_cuda, sizeof(float) * Size * Size);
    if (res != CUDA_SUCCESS) {
        printf("cuMemAlloc failed: res = %u\n", res);
        return -1;
    }

    res = cuMemAlloc(&a_cuda, sizeof(float) * Size * Size);
    if (res != CUDA_SUCCESS) {
        printf("cuMemAlloc failed: res = %u\n", res);
        return -1;
    }

    res = cuMemAlloc(&b_cuda, sizeof(float) * Size);
    if (res != CUDA_SUCCESS) {
        printf("cuMemAlloc failed: res = %u\n", res);
        return -1;
    }

    /* Copy data from main memory to device memory */
    res = cuMemcpyHtoD(a_cuda, a, sizeof(float) * Size * Size);
    if (res != CUDA_SUCCESS) {
        printf("cuMemcpyHtoD failed: res = %u\n", res);
        return -1;
    }

    res = cuMemcpyHtoD(b_cuda, b, sizeof(float) * Size);
    if (res != CUDA_SUCCESS) {
        printf("cuMemcpyHtoD failed: res = %u\n", res);
        return -1;
    }

    res = cuMemcpyHtoD(m_cuda, m, sizeof(float) * Size * Size);
    if (res != CUDA_SUCCESS) {
        printf("cuMemcpyHtoD failed: res = %u\n", res);
        return -1;
    }

    int block_size, grid_size;
    block_size = MAXBLOCKSIZE;
    grid_size = (Size/block_size) + (!(Size%block_size)? 0:1);
    //printf("1d grid size: %d\n",grid_size);

    int blockSize2d, gridSize2d;
    blockSize2d = 4;
    gridSize2d = (Size/blockSize2d) + (!(Size%blockSize2d?0:1)); 

    // begin timing kernels
    struct timeval time_start;
    gettimeofday(&time_start, NULL);

    // run kernels
    for (t=0; t<(Size-1); t++) {
        gaussian_launch(mod, gridSize2d, gridSize2d, blockSize2d, blockSize2d, m_cuda, a_cuda, Size, t);
        gaussian_launch2(mod, gridSize2d, gridSize2d, blockSize2d, blockSize2d, m_cuda, a_cuda, b_cuda, Size, Size-t, t);
    }

    // end timing kernels
    struct timeval time_end;
    gettimeofday(&time_end, NULL);
    totalKernelTime = (time_end.tv_sec * 1000000 + time_end.tv_usec) - (time_start.tv_sec * 1000000 + time_start.tv_usec);

    /*
    // copy memory back to CPU
    cudaMemcpy(m, m_cuda, Size * Size * sizeof(float),cudaMemcpyDeviceToHost );
    cudaMemcpy(a, a_cuda, Size * Size * sizeof(float),cudaMemcpyDeviceToHost );
    cudaMemcpy(b, b_cuda, Size * sizeof(float),cudaMemcpyDeviceToHost );
    cudaFree(m_cuda);
    cudaFree(a_cuda);
    cudaFree(b_cuda);
    */
}

/*------------------------------------------------------
 ** BackSub() -- Backward substitution
 **------------------------------------------------------
 */

void BackSub()
{
    // create a new vector to hold the final answer
    finalVec = (float *) malloc(Size * sizeof(float));
    // solve "bottom up"
    int i,j;
    for(i=0;i<Size;i++){
        finalVec[Size-i-1]=b[Size-i-1];
        for(j=0;j<i;j++)
        {
            finalVec[Size-i-1]-=*(a+Size*(Size-i-1)+(Size-j-1)) * finalVec[Size-j-1];
        }
        finalVec[Size-i-1]=finalVec[Size-i-1]/ *(a+Size*(Size-i-1)+(Size-i-1));
    }
}
