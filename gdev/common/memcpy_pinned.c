#include <cuda.h>
#include <string.h>
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

int cuda_test_memcpy_pinned(unsigned int size)
{
	int i;
	CUresult res;
	CUdevice dev;
	CUcontext ctx;
	CUdeviceptr data_addr;
	unsigned int *buf, *pin;
	struct timeval tv;
	struct timeval tv_total_start, tv_total_end;
	unsigned long total;
	struct timeval tv_h2d_start, tv_h2d_end;
	unsigned long h2d;
	struct timeval tv_d2h_start, tv_d2h_end;
	unsigned long d2h;
	struct timeval tv_mem_alloc_start;
	float init_gpu, close_gpu, mem_alloc, hostcpy;

	buf = malloc(size);
	if (!buf) {
		printf("malloc failed\n");
		return -1;
	}

	res = cuMemAllocHost((void**) &pin, size);
	if (res != CUDA_SUCCESS) {
		printf("cuMemAllocHost failed: res = %u\n", (unsigned int)res);
		return -1;
	}

	for (i = 0; i < size / 4; i++) {
		pin[i] = i+1;
	}

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

	gettimeofday(&tv_mem_alloc_start, NULL);
	res = cuMemAlloc(&data_addr, size);
	if (res != CUDA_SUCCESS) {
		printf("cuMemAlloc failed: res = %u\n", (unsigned int)res);
		return -1;
	}

	gettimeofday(&tv_h2d_start, NULL);
	res = cuMemcpyHtoD(data_addr, pin, size);
	gettimeofday(&tv_h2d_end, NULL);

	if (res != CUDA_SUCCESS) {
		printf("cuMemcpyHtoD failed: res = %u\n", (unsigned int)res);
		return -1;
	}

	cuCtxSynchronize();

	memcpy(buf, pin, size);
	memset(pin, size, 0);

	gettimeofday(&tv_d2h_start, NULL);
	res = cuMemcpyDtoH(pin, data_addr, size);
	gettimeofday(&tv_d2h_end, NULL);

	if (res != CUDA_SUCCESS) {
		printf("cuMemcpyDtoH failed: res = %u\n", (unsigned int)res);
		return -1;
	}

	res = cuMemFree(data_addr);
	if (res != CUDA_SUCCESS) {
		printf("cuMemFree failed: res = %u\n", (unsigned int)res);
		return -1;
	}


	res = cuCtxDestroy(ctx);
	if (res != CUDA_SUCCESS) {
		printf("cuCtxDestroy failed: res = %u\n", (unsigned int)res);
		return -1;
	}

	gettimeofday(&tv_total_end, NULL);

	res = cuMemFreeHost(pin);
	if (res != CUDA_SUCCESS) {
		printf("cuMemFreeHost failed: res = %u\n", (unsigned int)res);
		return -1;
	}

#if 0
	for (i = 0; i < size / 4; i++) {
		if (pin[i] != buf[i]) {
			printf("pin[%d] = %u, buf[%d] = %u\n",
				   i, pin[i], i, buf[i]);
			goto end;
		}
	}
#endif

	tvsub(&tv_mem_alloc_start, &tv_total_start, &tv);
	init_gpu = tv.tv_sec * 1000.0 + (float) tv.tv_usec / 1000.0;
	tvsub(&tv_h2d_start, &tv_mem_alloc_start, &tv);
	mem_alloc = tv.tv_sec * 1000.0 + (float) tv.tv_usec / 1000.0;
	tvsub(&tv_h2d_end, &tv_h2d_start, &tv);
	h2d = tv.tv_sec * 1000 + tv.tv_usec / 1000;
	tvsub(&tv_d2h_start, &tv_h2d_end, &tv);
	hostcpy = tv.tv_sec * 1000 + tv.tv_usec / 1000;
	tvsub(&tv_d2h_end, &tv_d2h_start, &tv);
	d2h = tv.tv_sec * 1000 + tv.tv_usec / 1000;
	tvsub(&tv_total_end, &tv_d2h_end, &tv);
	close_gpu = tv.tv_sec * 1000.0 + (float) tv.tv_usec / 1000.0;
	tvsub(&tv_total_end, &tv_total_start, &tv);
	total = tv.tv_sec * 1000.0 + (float) tv.tv_usec / 1000.0;

	printf("Init: %f\n", init_gpu);
	printf("MemAlloc: %f\n", mem_alloc);
	printf("HtoD: %f\n", h2d);
	printf("HostCpy: %f\n", hostcpy);
	printf("DtoH: %f\n", d2h);
	printf("Close: %f\n", close_gpu);
	printf("Total: %f\n", total);

	free(buf);

	return 0;

end:
	free(buf);

	return -1;
}
