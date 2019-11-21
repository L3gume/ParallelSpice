
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <math.h>
#include <vector>
#include <string>
#include <iostream>
#include "JsonParser.h"

cudaError_t solve(json::parse_result& data);
void choleskyDecomposition(double* A, double* L, int dim);

//__global__ void choleskyTimepointSolverKernel(double* A, double* L, double* bs, double* y, double* xs, int dim) {
//	const auto time_point = blockIdx.x * blockDim.x + threadIdx.x;//	const auto b = &bs[time_point * dim];//	auto accumulator = 0.0;////	// Reset y to 0//	for (auto i = 0; i < dim; i++)//	{//		y[i] = 0.0;//	}////	// Solve Ly = b//	for (auto i = 0; i < dim; i++)//	{//		accumulator = 0.0;////		for (auto j = 0; j < i; j++)//		{//			accumulator += L[i * dim + j] * y[j];//		}////		y[i] = (b[i] - accumulator) / L[i * dim + i];//	}//
//	// Solve (L_T)x = y
//	for (auto i = dim - 1; i < 0; i--)
//	{
//		accumulator = 0.0;
//		for (auto j = i + 1; j < dim; j++)
//		{
//			accumulator += L[j * dim + i] * xs[time_point * dim + j];
//		}
//		
//		xs[time_point * dim + i] = (y[i] - accumulator) / L[i * dim + i];
//	}
//}

template <int N, int M>
__global__ void test_kernel(cuda_mem::cuda_unique_2d<float>* ptr1, cuda_mem::cuda_unique_ptr<float>* ptr2, const int n_threads) {
    const auto idx = blockIdx.x * blockDim.x * threadIdx.x;
    (*ptr2)[idx] = 69.f;
    
    const auto start_i = idx * N;
    auto n_i = N / n_threads;
    const auto i_extra = N % n_threads;
    if (idx == n_threads - 1) n_i += i_extra;
    
    const auto start_j = idx * M;
    auto n_j = M / n_threads;
    const auto j_extra = M % n_threads;
    if (idx == n_threads - 1) n_j += j_extra;
    
    for (auto i = start_i; i < start_i + n_i; ++i) {
        for (auto j = start_j; j < start_j + n_j; ++j) {
            ptr1->at(i, j) = 69.f;
        }
    }
}

// Perform Cholesky decomposition
void choleskyDecomposition(double* A, double* L, int dim) {
	auto accumulator = 0.0;

	for (auto i = 0; i < dim; i++)
	{
		for (auto j = 0; j < (i + 1); j++)
		{
			accumulator = 0.0;
			for (auto k = 0; k < j; k++)
			{
				accumulator += L[i * dim + k] * L[j * dim + k];
			}

			if (i == j)
			{
				L[i * dim + j] = sqrt(A[i * dim + i] - accumulator);
			}
			else
			{
				L[i * dim + j] = (1.0 / L[j * dim + j]) * (A[i * dim + j] - accumulator);
			}
		}
	}
}

int main(const int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr <<
        R"(Invalid number of arguments! usage:
        parallelSpice.exe [json file path])" << '\n';
        exit(1);
    }
    const auto file_path = std::string{argv[1]};
    auto result = json::parse_result{};
    auto parser = json::JsonParser(file_path);
    if (!parser.parse(result)) {
        exit(1);
    }
    
    // TESTING A THING
    auto g = cuda_mem::make_grid<float>(3, 3, 0.f);
    auto v = std::vector<float>{};
    v.assign(9, 0.f);
    auto g_device = cuda_mem::grid_to_cuda_2d(g);
    auto v_device = cuda_mem::make_cuda_unique(v);
    test_kernel<3, 3> <<<1, 3>>> (&g_device, &v_device, 3);
    v = cuda_mem::cuda_unique_to_vector(9, v_device);
    g = cuda_mem::cuda_2d_to_grid(3, 3, g_device);
    // END OF TESTING A THING
    
    
    // Solve.
    auto cudaStatus = solve(result);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "solve failed!");
        return 1;
    }

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

cudaError_t solve(json::parse_result& data) {
	const auto dim = 3;
	const auto num_timepoints = 10;

	std::vector<double> A = { 25, 15, -5,
							  15, 18,  0,
							  -5,  0, 11 };
	std::vector<double> L(dim * dim, 0.0);
	std::vector<double> Bs(num_timepoints * dim, 10.0);
	std::vector<double> Xs(num_timepoints * dim, 0.0);
	std::vector<double> y(dim, 0.0);

	double* dev_A = 0;
	double* dev_L = 0;
	double* dev_Bs = 0;
	double* dev_Xs = 0;
	double* dev_Y = 0;

	// Perform Cholesky decomposition
	choleskyDecomposition(A.data(), L.data(), dim);

	// Choose which GPU to run on, change this on a multi-GPU system.
	auto cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for image and new image.
	cudaStatus = cudaMalloc((void**)& dev_A, dim * dim * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc for A array failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)& dev_L, dim * dim * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc for L array failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)& dev_Bs, num_timepoints * dim * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc for Bs array failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)& dev_Xs, num_timepoints * dim * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc for Xs array failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)& dev_Y, dim * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc for Y array failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_A, A.data(), dim * dim * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy from A to dev_A failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_L, L.data(), dim * dim * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy from L to dev_L failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_Bs, Bs.data(), num_timepoints * dim * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy from Bs to dev_Bs failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_Xs, Xs.data(), num_timepoints * dim * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy from Xs to dev_Xs failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_Y, y.data(), dim * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy from Y to dev_Y failed!");
		goto Error;
	}

	// Perform solver for each timepoint
	//choleskyTimepointSolverKernel << <1, num_timepoints >> > (dev_A, dev_L, dev_Bs, dev_Y, dev_Xs, dim);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "choleskyTimepointSolverKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching choleskyTimepointSolverKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(Xs.data(), dev_Xs, num_timepoints * dim * sizeof(double), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy from dev_Xs to Xs failed!");
		goto Error;
	}

	// Print timepoints
	for (auto i = 0; i < num_timepoints; i++)
	{
		printf("Timepoint %d", i + 1);
		for (auto j = 0; j < dim; j++)
		{
			printf("%lf ", Xs[i * dim + j]);
		}
		printf("\n");
	}

Error:
	cudaFree(dev_A);
	cudaFree(dev_L);
	cudaFree(dev_Bs);
	cudaFree(dev_Xs);

	return cudaStatus;
}