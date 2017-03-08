#include <cuda.h>
#ifdef __KERNEL__ /* just for measurement */
#include <linux/vmalloc.h>
#include <linux/time.h>
#define printf printk
#define malloc vmalloc
#define free vfree
#define gettimeofday(x, y) do_gettimeofday(x)
#else /* just for measurement */
#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#endif

/* tvsub: ret = x - y. */
static inline void tvsub(struct timeval *x, 
						 struct timeval *y, 
						 struct timeval *ret)
{
	ret->tv_sec = x->tv_sec - y->tv_sec;
	ret->tv_usec = x->tv_usec - y->tv_usec;
	if (ret->tv_usec < 0) {
		ret->tv_sec--;
		ret->tv_usec += 1000000;
	}
}

int cuda_test_memcpy_async(unsigned int size)
{
	int i;
	CUresult res;
	CUdevice dev;
	CUcontext ctx;
	CUstream stream;
	CUdeviceptr data_addr;
	unsigned int *in, *out;
	struct timeval tv;
	struct timeval tv_total_start, tv_total_end;
    float total;
	struct timeval tv_h2d_start, tv_h2d_end;
	float h2d;
	struct timeval tv_d2h_start, tv_d2h_end;
	float d2h;
	struct timeval tv_mem_alloc_start, tv_init_start;
	float init_gpu, close_gpu, mem_alloc, data_init;

	gettimeofday(&tv_total_start, NULL);

	res = cuInit(0);
	if (res != CUDA_SUCCESS) {
		printf("cuInit failed: res = %u\n", (unsigned int)res);
		return -1;
	}

	res = cuDeviceGet(&dev, 0);
	if (res != CUDA_SUCCESS) {
		printf("cuDeviceGet failed: res = %u\n", (unsigned int)res);
		return -1;
	}

	res = cuCtxCreate(&ctx, 0, dev);
	if (res != CUDA_SUCCESS) {
		printf("cuCtxCreate failed: res = %u\n", (unsigned int)res);
		return -1;
	}

	res = cuStreamCreate(&stream, 0);
	if (res != CUDA_SUCCESS) {
		printf("cuStreamCreate failed: res = %u\n", (unsigned int)res);
		return -1;
	}

	gettimeofday(&tv_init_start, NULL);
	res = cuMemAllocHost((void **)&in, size);
	if (res != CUDA_SUCCESS) {
		printf("cuMemAllocHost(in) failed: res = %u\n", (unsigned int)res);
		return -1;
	}

	res = cuMemAllocHost((void **)&out, size);
	if (res != CUDA_SUCCESS) {
		printf("cuMemAllocHost(out) failed: res = %u\n", (unsigned int)res);
		return -1;
	}

	for (i = 0; i < size / 4; i++) {
		in[i] = i+1;
		out[i] = 0;
	}

	gettimeofday(&tv_mem_alloc_start, NULL);
	res = cuMemAlloc(&data_addr, size);
	if (res != CUDA_SUCCESS) {
		printf("cuMemAlloc failed: res = %u\n", (unsigned int)res);
		return -1;
	}

	gettimeofday(&tv_h2d_start, NULL);
	res = cuMemcpyHtoDAsync(data_addr, in, size, stream);
	if (res != CUDA_SUCCESS) {
		printf("cuMemcpyHtoDAsync failed: res = %u\n", (unsigned int)res);
		return -1;
	}
	res = cuStreamSynchronize(stream);
	if (res != CUDA_SUCCESS) {
		printf("cuStreamSynchronize() failed: res = %u\n", (unsigned int)res);
		return -1;
	}
	gettimeofday(&tv_h2d_end, NULL);

	gettimeofday(&tv_d2h_start, NULL);
	res = cuMemcpyDtoHAsync(out, data_addr, size, stream);
	if (res != CUDA_SUCCESS) {
		printf("cuMemcpyDtoHAsync failed: res = %u\n", (unsigned int)res);
		return -1;
	}
	res = cuStreamSynchronize(stream);
	if (res != CUDA_SUCCESS) {
		printf("cuStreamSynchronize() failed: res = %u\n", (unsigned int)res);
		return -1;
	}
	gettimeofday(&tv_d2h_end, NULL);

	res = cuMemFree(data_addr);
	if (res != CUDA_SUCCESS) {
		printf("cuMemFree failed: res = %u\n", (unsigned int)res);
		return -1;
	}

	res = cuStreamDestroy(stream);
	if (res != CUDA_SUCCESS) {
		printf("cuStreamDestroy failed: res = %u\n", (unsigned int)res);
		return -1;
	}

	res = cuCtxDestroy(ctx);
	if (res != CUDA_SUCCESS) {
		printf("cuCtxDestroy failed: res = %u\n", (unsigned int)res);
		return -1;
	}

	gettimeofday(&tv_total_end, NULL);

	for (i = 0; i < size / 4; i++) {
		if (in[i] != out[i]) {
			printf("in[%d] = %u, out[%d] = %u\n",
				   i, in[i], i, out[i]);
		}
	}

	res = cuMemFreeHost(out);
	if (res != CUDA_SUCCESS) {
		printf("cuMemFreeHost(out) failed: res = %u\n", (unsigned int)res);
		return -1;
	}

	res = cuMemFreeHost(in);
	if (res != CUDA_SUCCESS) {
		printf("cuMemFreeHost(in) failed: res = %u\n", (unsigned int)res);
		return -1;
	}

	tvsub(&tv_init_start, &tv_total_start, &tv);
	init_gpu = tv.tv_sec * 1000.0 + (float) tv.tv_usec / 1000.0;
	tvsub(&tv_mem_alloc_start, &tv_init_start, &tv);
	data_init = tv.tv_sec * 1000.0 + (float) tv.tv_usec / 1000.0;
	tvsub(&tv_h2d_start, &tv_mem_alloc_start, &tv);
	mem_alloc = tv.tv_sec * 1000.0 + (float) tv.tv_usec / 1000.0;
	tvsub(&tv_h2d_end, &tv_h2d_start, &tv);
	h2d = tv.tv_sec * 1000 + tv.tv_usec / 1000;
	tvsub(&tv_d2h_end, &tv_d2h_start, &tv);
	d2h = tv.tv_sec * 1000 + tv.tv_usec / 1000;
	tvsub(&tv_total_end, &tv_d2h_end, &tv);
	close_gpu = tv.tv_sec * 1000.0 + (float) tv.tv_usec / 1000.0;
	tvsub(&tv_total_end, &tv_total_start, &tv);
	total = tv.tv_sec * 1000.0 + (float) tv.tv_usec / 1000.0;

	printf("Init: %f\n", init_gpu);
	printf("DataInit: %f\n", data_init);
	printf("MemAlloc: %f\n", mem_alloc);
	printf("HtoD: %f\n", h2d);
	printf("DtoH: %f\n", d2h);
	printf("Close: %f\n", close_gpu);
	printf("Total: %f\n", total);

	return 0;

end:

	return -1;
}
