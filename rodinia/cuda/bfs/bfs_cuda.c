/*
 * Implementing Breadth first search on CUDA using algorithm given in HiPC'07
 * paper "Accelerating Large Graph Algorithms on the GPU using CUDA"
 *
 * Copyright (c) 2008 
 * International Institute of Information Technology - Hyderabad. 
 * All rights reserved.
 *
 * Permission to use, copy, modify and distribute this software and its 
 * documentation for educational purpose is hereby granted without fee, 
 * provided that the above copyright notice and this permission notice 
 * appear in all copies of this software and that you do not sell the software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND WITHOUT WARRANTY OF ANY KIND,EXPRESS, 
 * IMPLIED OR OTHERWISE.
 *
 * Created by Pawan Harish.
 *
 * Modified by Shinpei Kato.
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda.h>
#include "util.h" /* cuda_driver_api_{init,exit}() */
#include "bfs.h"

int no_of_nodes;
int edge_list_size;
FILE *fp;

struct timeval tv;
struct timeval tv_total_start, tv_total_end;
float total;
struct timeval tv_h2d_start, tv_h2d_end;
float h2d;
struct timeval tv_d2h_start, tv_d2h_end;
float d2h = 0;
struct timeval tv_exec_start, tv_exec_end;
struct timeval tv_mem_alloc_start;
struct timeval tv_close_start;
float mem_alloc;
float exec = 0;
float init_gpu;
float close_gpu;

int bfs_launch
(CUmodule mod, int nr_blocks, int nr_threads_per_block, int nr_nodes,
 CUdeviceptr d_over, CUdeviceptr d_graph_nodes, CUdeviceptr d_graph_edges, 
 CUdeviceptr d_graph_mask, CUdeviceptr d_updating_graph_mask, 
 CUdeviceptr d_graph_visited, CUdeviceptr d_cost)
{
	int bdx, bdy, gdx, gdy;
	int k = 0;
	int stop;
	CUfunction f1, f2;
	CUresult res;

	bdx = nr_threads_per_block;
	bdy = 1;
	gdx = nr_blocks;
	gdy = 1;

	/* get functions. */
	res = cuModuleGetFunction(&f1, mod, "_Z6KernelP4NodePiS1_S1_S1_S1_i");
	if (res != CUDA_SUCCESS) {
		printf("cuModuleGetFunction(f1) failed: res = %u\n", res);
		return -1;
	}
	res = cuModuleGetFunction(&f2, mod, "_Z7Kernel2PiS_S_S_i");
	if (res != CUDA_SUCCESS) {
		printf("cuModuleGetFunction(f2) failed: res = %u\n", res);
		return -1;
	}

	/* Call the Kernel untill all the elements of Frontier are not false */
	do {
		/* if no thread changes this value then the loop stops */
	    gettimeofday(&tv_h2d_start, NULL);
		stop = false;
		res = cuMemcpyHtoD(d_over, &stop, sizeof(int));
		if (res != CUDA_SUCCESS) {
			printf("cuMemcpyHtoD(d_over) failed\n");
			return -1;
		}
	    gettimeofday(&tv_h2d_end, NULL);
		tvsub(&tv_h2d_end, &tv_h2d_start, &tv);
	    h2d += tv.tv_sec * 1000.0 + (float) tv.tv_usec / 1000.0;

	    /* f1 */
		void *param1[] = {&d_graph_nodes, &d_graph_edges, &d_graph_mask, 
						  &d_updating_graph_mask, &d_graph_visited, &d_cost,
						  &nr_nodes};
		res = cuLaunchKernel(f1, gdx, gdy, 1, bdx, bdy, 1, 0, 0, 
							 (void**)param1, NULL);
        if (res != CUDA_SUCCESS) {
            printf("cuLaunchKernel(f1) failed: res = %u\n", res);
            return -1;
        }
		cuCtxSynchronize();
		/* check if kernel execution generated and error */

		/* f2 */
		void *param2[] = {&d_graph_mask, &d_updating_graph_mask, 
						  &d_graph_visited, &d_over,  &nr_nodes};
		res = cuLaunchKernel(f2, gdx, gdy, 1, bdx, bdy, 1, 0, 0, 
							 (void**)param2, NULL);
        if (res != CUDA_SUCCESS) {
            printf("cuLaunchKernel(f2) failed: res = %u\n", res);
            return -1;
        }
		/* check if kernel execution generated and error */
	    gettimeofday(&tv_exec_end, NULL);
		tvsub(&tv_exec_end, &tv_h2d_end, &tv);
		exec += tv.tv_sec * 1000.0 + (float) tv.tv_usec / 1000.0;

		res = cuMemcpyDtoH(&stop, d_over, sizeof(int));
		if (res != CUDA_SUCCESS) {
			printf("cuMemcpyDtoH(stop) failed: res = %u\n", res);
			return -1;
		}
	    gettimeofday(&tv_d2h_end, NULL);
		tvsub(&tv_d2h_end, &tv_exec_end, &tv);
	    d2h += tv.tv_sec * 1000.0 + (float) tv.tv_usec / 1000.0;

		cuCtxSynchronize();
		k++;
	} while (stop);

	return 0;
}

void Usage(int argc, char**argv)
{
	fprintf(stderr,"Usage: %s <input_file>\n", argv[0]);
}

int BFSGraph(int argc, char** argv) 
{
	char *input_f;
	int source = 0;
	int num_of_blocks = 1;
	int num_of_threads_per_block = no_of_nodes;
	struct timeval tv;
	struct Node *h_graph_nodes;
	int *h_graph_mask;
	int *h_updating_graph_mask;
	int *h_graph_visited;
	int *h_graph_edges;
	int *h_cost;
	int start, edgeno;   
	int id, cost;
	int i;
	CUdeviceptr d_graph_nodes;
	CUdeviceptr d_graph_edges;
	CUdeviceptr d_graph_mask;
	CUdeviceptr d_updating_graph_mask;
	CUdeviceptr d_graph_visited;
	CUdeviceptr d_cost;
	CUdeviceptr d_over;
	CUcontext ctx;
	CUmodule mod;
	CUresult res;

	if (argc != 2) {
		Usage(argc, argv);
		exit(0);
	}
	
	input_f = argv[1];
	fp = fopen(input_f, "r");
	if (!fp) {
		printf("Error Reading graph file\n");
		return -1;
	}

	fscanf(fp, "%d", &no_of_nodes);

	/* Make execution Parameters according to the number of nodes and 
	   distribute threads across multiple Blocks if necessary */
	if (no_of_nodes > MAX_THREADS_PER_BLOCK) {
		num_of_blocks = (int)ceil(no_of_nodes / (double)MAX_THREADS_PER_BLOCK); 
		num_of_threads_per_block = MAX_THREADS_PER_BLOCK; 
	}

	/* allocate host memory */
	h_graph_nodes = (struct Node*) malloc(sizeof(struct Node) * no_of_nodes);
	h_graph_mask = (int*) malloc(sizeof(int) * no_of_nodes);
	h_updating_graph_mask = (int*) malloc(sizeof(int) * no_of_nodes);
	h_graph_visited = (int*) malloc(sizeof(int) * no_of_nodes);

	/* initalize the memory */
	for(i = 0; i < no_of_nodes; i++)  {
		fscanf(fp, "%d %d", &start, &edgeno);
		h_graph_nodes[i].starting = start;
		h_graph_nodes[i].no_of_edges = edgeno;
		h_graph_mask[i]=false;
		h_updating_graph_mask[i]=false;
		h_graph_visited[i]=false;
	}

	/* read the source node from the file */
	fscanf(fp, "%d", &source);
	source = 0;

	/* set the source node as true in the mask */
	h_graph_mask[source] = true;
	h_graph_visited[source] = true;

	fscanf(fp, "%d", &edge_list_size);

	h_graph_edges = (int*) malloc(sizeof(int) * edge_list_size);
	for(i = 0; i < edge_list_size ; i++) {
		fscanf(fp, "%d", &id);
		fscanf(fp, "%d", &cost);
		h_graph_edges[i] = id;
	}

	if (fp)
		fclose(fp);    

	/* allocate mem for the result on host side */
	h_cost = (int*) malloc(sizeof(int) * no_of_nodes);
	for(i = 0; i < no_of_nodes; i++)
		h_cost[i] = -1;
	h_cost[source] = 0;

	/*
	 * call our common CUDA initialization utility function.
	 */
	gettimeofday(&tv_total_start, NULL);
	res = cuda_driver_api_init(&ctx, &mod, "./bfs.cubin");
	if (res != CUDA_SUCCESS) {
		printf("cuda_driver_api_init failed: res = %u\n", res);
		return -1;
	}

	gettimeofday(&tv_mem_alloc_start, NULL);
	tvsub(&tv_mem_alloc_start, &tv_total_start, &tv);
	init_gpu = tv.tv_sec * 1000.0 + (float) tv.tv_usec / 1000.0;

	/*
	 * allocate device memory space
	 */
	res = cuMemAlloc(&d_graph_nodes, sizeof(struct Node) * no_of_nodes);
	if (res != CUDA_SUCCESS) {
		printf("cuMemAlloc failed: res = %u\n", res);
		return -1;
	}
	res = cuMemAlloc(&d_graph_edges, sizeof(int) * edge_list_size);
	if (res != CUDA_SUCCESS) {
		printf("cuMemAlloc failed: res = %u\n", res);
		return -1;
	}
	res = cuMemAlloc(&d_graph_mask, sizeof(int) * no_of_nodes);
	if (res != CUDA_SUCCESS) {
		printf("cuMemAlloc failed: res = %u\n", res);
		return -1;
	}
	res = cuMemAlloc(&d_updating_graph_mask, sizeof(int) * no_of_nodes);
	if (res != CUDA_SUCCESS) {
		printf("cuMemAlloc failed: res = %u\n", res);
		return -1;
	}
	res = cuMemAlloc(&d_graph_visited, sizeof(int) * no_of_nodes);
	if (res != CUDA_SUCCESS) {
		printf("cuMemAlloc failed: res = %u\n", res);
		return -1;
	}
	res = cuMemAlloc(&d_cost, sizeof(int) * no_of_nodes);
	if (res != CUDA_SUCCESS) {
		printf("cuMemAlloc failed: res = %u\n", res);
		return -1;
	}
	/* make a int to check if the execution is over */
	res = cuMemAlloc(&d_over, sizeof(int));
	if (res != CUDA_SUCCESS) {
		printf("cuMemAlloc failed: res = %u\n", res);
		return -1;
	}

	gettimeofday(&tv_h2d_start, NULL);
	tvsub(&tv_h2d_start, &tv_mem_alloc_start, &tv);
	mem_alloc = tv.tv_sec * 1000.0 + (float) tv.tv_usec / 1000.0;

	/* copy the node list to device memory */
	res = cuMemcpyHtoD(d_graph_nodes, h_graph_nodes, sizeof(struct Node) * no_of_nodes);
	if (res != CUDA_SUCCESS) {
		printf("cuMemcpyHtoD failed: res = %u\n", res);
		return -1;
	}
	/* copy the edge List to device memory */
	res = cuMemcpyHtoD(d_graph_edges, h_graph_edges, sizeof(int) * edge_list_size);
	if (res != CUDA_SUCCESS) {
		printf("cuMemcpyHtoD failed: res = %u\n", res);
		return -1;
	}
	/* copy the mask to device memory */
	res = cuMemcpyHtoD(d_graph_mask, h_graph_mask, sizeof(int) * no_of_nodes);
	if (res != CUDA_SUCCESS) {
		printf("cuMemcpyHtoD failed: res = %u\n", res);
		return -1;
	}
	res = cuMemcpyHtoD(d_updating_graph_mask, h_updating_graph_mask, sizeof(int) * no_of_nodes);
	if (res != CUDA_SUCCESS) {
		printf("cuMemcpyHtoD failed: res = %u\n", res);
		return -1;
	}
	/* copy the visited nodes array to device memory */
	res = cuMemcpyHtoD(d_graph_visited, h_graph_visited, sizeof(int) * no_of_nodes);
	if (res != CUDA_SUCCESS) {
		printf("cuMemcpyHtoD failed: res = %u\n", res);
		return -1;
	}
	/* device memory for result */
	res = cuMemcpyHtoD(d_cost, h_cost, sizeof(int) * no_of_nodes);
	if (res != CUDA_SUCCESS) {
		printf("cuMemcpyHtoD failed: res = %u\n", res);
		return -1;
	}
	gettimeofday(&tv_h2d_end, NULL);
	tvsub(&tv_h2d_end, &tv_h2d_start, &tv);
	h2d = tv.tv_sec * 1000.0 + (float) tv.tv_usec / 1000.0;

	/* we cannot use maximum thread number because of the virtualization
 	 * limitation in a3.h */
	bfs_launch(mod, num_of_blocks, num_of_threads_per_block / 4, no_of_nodes,
			   d_over, d_graph_nodes, d_graph_edges, d_graph_mask, 
			   d_updating_graph_mask, d_graph_visited, d_cost);

	/* copy result from device to host */
	gettimeofday(&tv_d2h_start, NULL);
	res = cuMemcpyDtoH(h_cost, d_cost, sizeof(int) * no_of_nodes);
	if (res != CUDA_SUCCESS) {
		printf("cuMemcpyDtoH failed: res = %u\n", res);
		return -1;
	}
	gettimeofday(&tv_d2h_end, NULL);
    tvsub(&tv_d2h_end, &tv_d2h_start, &tv);
	d2h += tv.tv_sec * 1000.0 + (float) tv.tv_usec / 1000.0;

	gettimeofday(&tv_close_start, NULL);
	/* cleanup memory */
	cuMemFree(d_graph_nodes);
	cuMemFree(d_graph_edges);
	cuMemFree(d_graph_mask);
	cuMemFree(d_updating_graph_mask);
	cuMemFree(d_graph_visited);
	cuMemFree(d_cost);
	cuMemFree(d_over);

	res = cuda_driver_api_exit(ctx, mod);
	if (res != CUDA_SUCCESS) {
		printf("cuda_driver_api_exit faild: res = %u\n", res);
		return -1;
	}
	gettimeofday(&tv_total_end, NULL);

	/* Store the result into a file */
#if 0
	{
		FILE *fpo = fopen("result.txt", "w");
		for(i = 0; i < no_of_nodes; i++)
			fprintf(fpo, "%d) cost:%d\n", i, h_cost[i]);
		fclose(fpo);
		printf("/* Result stored in result.txt */\n");
	}
#endif

	free(h_graph_nodes);
	free(h_graph_edges);
	free(h_graph_mask);
	free(h_updating_graph_mask);
	free(h_graph_visited);
	free(h_cost);

	tvsub(&tv_total_end, &tv_close_start, &tv);
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
	return 0;
}

int main( int argc, char** argv) 
{
	no_of_nodes = 0;
	edge_list_size = 0;

	BFSGraph(argc, argv);

	return 0;
}
