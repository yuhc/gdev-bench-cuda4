// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda.h>

#include "util.h"
#include "backprop.h"

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
struct timeval tv_idle_start, tv_idle_end;
float mem_alloc;
float exec;
float init_gpu;
float close_gpu;
float idle;

////////////////////////////////////////////////////////////////////////////////

void bpnn_layerforward(float *l1, float *l2, float **conn, int n1, int n2);

void bpnn_output_error(float *delta, float *target, float *output, int nj, float *err);

void bpnn_hidden_error(float *delta_h, int nh, float *delta_o, int no, float **who, float *hidden, float *err);

void bpnn_adjust_weights(float *delta, int ndelta, float *ly, int nly, float **w, float **oldw);

float **alloc_2d_dbl(int m, int n);

float squash(float x);

unsigned int num_threads = 0;
unsigned int num_blocks = 0;

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) 
{
	setup(argc, argv);

	return 0;
}

CUresult bpnn_layerforward_launch
(CUmodule mod, CUdeviceptr input_cuda, CUdeviceptr output_hidden_cuda,
 CUdeviceptr input_hidden_cuda, CUdeviceptr hidden_partial_sum,
 int in, int hid) 
{
	int bdx, bdy, gdx, gdy;
	void* param[] = {&input_cuda, &output_hidden_cuda, &input_hidden_cuda,
					 &hidden_partial_sum, &in, &hid};
	CUfunction f;
	CUresult res;

	bdx = 16;
	bdy = 16;
	gdx = 1;
	gdy = num_blocks;

	/* get functions. */
	res = cuModuleGetFunction(&f, mod, "_Z22bpnn_layerforward_CUDAPfS_S_S_ii");
	if (res != CUDA_SUCCESS) {
		printf("cuModuleGetFunction(layerforward) failed: res = %u\n", res);
		return res;
	}

	res = cuLaunchKernel(f, gdx, gdy, 1, bdx, bdy, 1, 0, 0, (void**) param, 0);
	if (res != CUDA_SUCCESS) {
		printf("cuLaunchKernel(layerforward) failed: res = %u\n", res);
		return res;
	}

	return CUDA_SUCCESS;
}

/* int is 64-bit for some reason... */
CUresult bpnn_adjust_weights_launch
(CUmodule mod, CUdeviceptr delta, long hid, CUdeviceptr ly, long in,          
 CUdeviceptr w, CUdeviceptr oldw) 
{
	int bdx, bdy, gdx, gdy;
	void* param[] = {&delta, &hid, &ly, &in, &w, &oldw};
	CUfunction f;
	CUresult res;

	bdx = 16;
	bdy = 16;
	gdx = 1;
	gdy = num_blocks;

	/* get functions. */
	res = cuModuleGetFunction(&f, mod, "_Z24bpnn_adjust_weights_cudaPfiS_iS_S_");
	if (res != CUDA_SUCCESS) {
		printf("cuModuleGetFunction(adjust_weights) failed: res = %u\n", res);
		return res;
	}

	res = cuLaunchKernel(f, gdx, gdy, 1, bdx, bdy, 1, 0, 0, (void**) param, 0);
	if (res != CUDA_SUCCESS) {
		printf("cuLaunchKernel(adjust_weights) failed: res = %u\n", res);
		return res;
	}

	return CUDA_SUCCESS;
}

void bpnn_train_cuda(BPNN *net, float *eo, float *eh)
{
	int j, k;
	int in, hid, out;
	float out_err, hid_err;

	in = net->input_n;
	hid = net->hidden_n;
	out = net->output_n;   

#ifdef GPU  
	int m = 0;
	float *partial_sum;
	float sum;
	float *input_weights_one_dim;
	float *input_weights_prev_one_dim;
	num_blocks = in / 16 / 32; // data size is enlarged 32 times in this test  
	CUdeviceptr input_cuda;
	CUdeviceptr input_hidden_cuda;
	CUdeviceptr output_hidden_cuda;
	CUdeviceptr hidden_partial_sum;
	CUdeviceptr hidden_delta_cuda;
	CUdeviceptr input_prev_weights_cuda;
	CUcontext ctx;
	CUmodule mod;
	CUresult res;

	input_weights_one_dim = 
		(float *) malloc((in + 1) * (hid + 1) * sizeof(float));
	input_weights_prev_one_dim = 
		(float *) malloc((in + 1) * (hid + 1) * sizeof(float));
	partial_sum = (float *) malloc(num_blocks * WIDTH * sizeof(float));

	/* this preprocessing stage is added to correct the bugs of wrong 
	   memcopy using two-dimensional net->inputweights */
	for (k = 0; k <= in; k++) {	
		for (j = 0; j <= hid; j++) {
			input_weights_one_dim[m] = net->input_weights[k][j];
			input_weights_prev_one_dim[m] = net-> input_prev_weights[k][j];
			m++;
		}
	}

	/*
	 * call our common CUDA initialization utility function.
	 */
	gettimeofday(&tv_total_start, NULL);
	res = cuda_driver_api_init(&ctx, &mod, "./backprop.cubin");
	if (res != CUDA_SUCCESS) {
		printf("cuda_driver_api_init failed: res = %u\n", res);
		return ;
	}

    gettimeofday(&tv_mem_alloc_start, NULL);
	tvsub(&tv_mem_alloc_start, &tv_total_start, &tv);
	init_gpu = tv.tv_sec * 1000.0 + (float) tv.tv_usec / 1000.0;

	/*
	 * allocate device memory space
	 */
	res = cuMemAlloc(&input_cuda, (in + 1) * sizeof(float));
	if (res != CUDA_SUCCESS) {
		printf("cuMemAlloc failed: res = %u\n", res);
		return ;
	}
	res = cuMemAlloc(&output_hidden_cuda, (hid + 1) * sizeof(float));
	if (res != CUDA_SUCCESS) {
		printf("cuMemAlloc failed: res = %u\n", res);
		return ;
	}
	res = cuMemAlloc(&input_hidden_cuda, (in + 1) * (hid + 1) * sizeof(float));
	if (res != CUDA_SUCCESS) {
		printf("cuMemAlloc failed: res = %u\n", res);
		return ;
	}
	res = cuMemAlloc(&hidden_partial_sum, num_blocks * WIDTH * sizeof(float));
	if (res != CUDA_SUCCESS) {
		printf("cuMemAlloc failed: res = %u\n", res);
		return ;
	}
	res = cuMemAlloc(&hidden_delta_cuda, (hid + 1) * sizeof(float));
	if (res != CUDA_SUCCESS) {
		printf("cuMemAlloc failed: res = %u\n", res);
		return ;
	}
	res = cuMemAlloc(&input_prev_weights_cuda, (in + 1) * (hid + 1) * sizeof(float));
	if (res != CUDA_SUCCESS) {
		printf("cuMemAlloc failed: res = %u\n", res);
		return ;
	}

    gettimeofday(&tv_h2d_start, NULL);
    tvsub(&tv_h2d_start, &tv_mem_alloc_start, &tv);
	mem_alloc = tv.tv_sec * 1000.0 + (float) tv.tv_usec / 1000.0;
#endif

#ifdef CPU
	printf("Performing CPU computation\n");
	bpnn_layerforward(net->input_units, net->hidden_units,net->input_weights, in, hid);
#endif

#ifdef GPU
	printf("Performing GPU computation\n");
    //printf("in= %d, hid = %d, numblocks = %d\n", in, hid, num_blocks);

	res = cuMemcpyHtoD(input_cuda, net->input_units, (in + 1) * sizeof(float));
	if (res != CUDA_SUCCESS) {
		printf("cuMemcpyHtoD failed: res = %u\n", res);
		return ;
	}
	res = cuMemcpyHtoD(input_hidden_cuda, input_weights_one_dim, (in + 1) * (hid + 1) * sizeof(float));
	if (res != CUDA_SUCCESS) {
		printf("cuMemcpyHtoD failed: res = %u\n", res);
		return ;
	}

    gettimeofday(&tv_h2d_end, NULL);
    tvsub(&tv_h2d_end, &tv_h2d_start, &tv);
    h2d = tv.tv_sec * 1000.0 + (float) tv.tv_usec / 1000.0;

	res = bpnn_layerforward_launch(mod, input_cuda, output_hidden_cuda,
								   input_hidden_cuda, hidden_partial_sum,
								   in, hid);
	if (res != CUDA_SUCCESS) {
		printf("bpnn_layerforward failed: res = %u\n", res);
		return ;
	}

	cuCtxSynchronize();

#if 0
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		printf("bpnn kernel error: %s\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}
#endif

    gettimeofday(&tv_exec_end, NULL);
    tvsub(&tv_exec_end, &tv_h2d_end, &tv);
    exec = tv.tv_sec * 1000.0 + (float) tv.tv_usec / 1000.0;

	res = cuMemcpyDtoH(partial_sum, hidden_partial_sum, num_blocks * WIDTH * sizeof(float));
	if (res != CUDA_SUCCESS) {
		printf("cuMemcpyDtoH(layerforward) failed: res = %u\n", res);
		return ;
	}

    gettimeofday(&tv_idle_start, NULL);

	for (j = 1; j <= hid; j++) {
		sum = 0.0;
		for (k = 0; k < num_blocks; k++) {	
			sum += partial_sum[k * hid + j-1] ;
		}
		sum += net->input_weights[0][j];
		net-> hidden_units[j] = (float) (1.0 / (1.0 + exp(-sum)));
	}
  #endif

	bpnn_layerforward(net->hidden_units, net->output_units, net->hidden_weights, hid, out);
	bpnn_output_error(net->output_delta, net->target, net->output_units, out, &out_err);
	bpnn_hidden_error(net->hidden_delta, hid, net->output_delta, out, net->hidden_weights, net->hidden_units, &hid_err);  
	bpnn_adjust_weights(net->output_delta, out, net->hidden_units, hid, net->hidden_weights, net->hidden_prev_weights);

#ifdef CPU
	bpnn_adjust_weights(net->hidden_delta, hid, net->input_units, in, net->input_weights, net->input_prev_weights);
#endif  

#ifdef GPU
    gettimeofday(&tv_idle_end, NULL);

	res = cuMemcpyHtoD(hidden_delta_cuda, net->hidden_delta, (hid + 1) * sizeof(float));
	if (res != CUDA_SUCCESS) {
		printf("cuMemcpyHtoD failed: res = %u\n", res);
		return ;
	}
	res = cuMemcpyHtoD(input_prev_weights_cuda, input_weights_prev_one_dim, (in + 1) * (hid + 1) * sizeof(float));
	if (res != CUDA_SUCCESS) {
		printf("cuMemcpyHtoD failed: res = %u\n", res);
		return ;
	}
	res = cuMemcpyHtoD(input_hidden_cuda, input_weights_one_dim, (in + 1) * (hid + 1) * sizeof(float));
	if (res != CUDA_SUCCESS) {
		printf("cuMemcpyHtoD failed: res = %u\n", res);
		return ;
	}

    gettimeofday(&tv_h2d_end, NULL);
    tvsub(&tv_h2d_end, &tv_idle_end, &tv);
    h2d += tv.tv_sec * 1000.0 + (float) tv.tv_usec / 1000.0;

	res = bpnn_adjust_weights_launch(mod, hidden_delta_cuda, hid, 
									 input_cuda, in, 
									 input_hidden_cuda, 
									 input_prev_weights_cuda);
	if (res != CUDA_SUCCESS) {
		printf("bpnn_adjust_weights failed: res = %u\n", res);
		return ;
	}

    gettimeofday(&tv_exec_end, NULL);
    tvsub(&tv_exec_end, &tv_h2d_end, &tv);
    exec += tv.tv_sec * 1000.0 + (float) tv.tv_usec / 1000.0;

	res = cuMemcpyDtoH(net->input_units, input_cuda, (in + 1) * sizeof(float));
	if (res != CUDA_SUCCESS) {
		printf("cuMemcpyDtoH(adjust_weights) failed: res = %u\n", res);
		return ;
	}

	res = cuMemcpyDtoH(input_weights_one_dim, input_hidden_cuda, (in + 1) * (hid + 1) * sizeof(float));
	if (res != CUDA_SUCCESS) {
		printf("cuMemcpyDtoH(adjust_weights) failed: res = %u\n", res);
		return ;
	}

	gettimeofday(&tv_d2h_end, NULL);
    tvsub(&tv_d2h_end, &tv_exec_end, &tv);
	d2h += tv.tv_sec * 1000.0 + (float) tv.tv_usec / 1000.0;

	cuMemFree(input_cuda);
	cuMemFree(output_hidden_cuda);
	cuMemFree(input_hidden_cuda);
	cuMemFree(hidden_partial_sum);
	cuMemFree(input_prev_weights_cuda);
	cuMemFree(hidden_delta_cuda);

	res = cuda_driver_api_exit(ctx, mod);
	if (res != CUDA_SUCCESS) {
		printf("cuda_driver_api_exit faild: res = %u\n", res);
		return ;
	}

	gettimeofday(&tv_total_end, NULL);
	tvsub(&tv_total_end, &tv_d2h_end, &tv);
	close_gpu = tv.tv_sec * 1000.0 + (float) tv.tv_usec / 1000.0;

	tvsub(&tv_idle_end, &tv_idle_start, &tv);
	idle = tv.tv_sec * 1000.0 + (float) tv.tv_usec / 1000.0;

	tvsub(&tv_total_end, &tv_total_start, &tv);
	total = tv.tv_sec * 1000.0 + (float) tv.tv_usec / 1000.0 - idle;

	printf("Init: %f\n", init_gpu);
	printf("MemAlloc: %f\n", mem_alloc);
	printf("HtoD: %f\n", h2d);
	printf("Exec: %f\n", exec);
	printf("DtoH: %f\n", d2h);
	printf("Close: %f\n", close_gpu);
	printf("Total: %f\n", total);

	free(partial_sum);
	free(input_weights_one_dim);
	free(input_weights_prev_one_dim);
#endif   
}
