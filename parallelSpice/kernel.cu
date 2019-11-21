
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <math.h>
#include <vector>
#include <string>
#include <iostream>
#include "JsonParser.h"
#include "dbg.h"

cudaError_t solve(json::parse_result& data);
void choleskyDecomposition(std::vector<double> A, std::vector<double> L, const int dim);
double* computeCoefficientMatrix(cuda_mem::grid<double> A, std::vector<double> Y, const int dim);

__global__ void choleskyTimepointSolverKernel(double* A, double* L, double* bs, double* y, double* xs, const int dim) {
	const auto time_point = blockIdx.x * blockDim.x + threadIdx.x;
	const auto b = &bs[time_point * dim];
	auto accumulator = 0.0;

	// Reset y to 0
	for (auto i = 0; i < dim; i++)
	{
		y[i] = 0.0;
	}

	// Solve Ly = b
	for (auto i = 0; i < dim; i++)
	{
		accumulator = 0.0;

		for (auto j = 0; j < i; j++)
		{
			accumulator += L[i * dim + j] * y[j];
		}

		y[i] = (b[i] - accumulator) / L[i * dim + i];
	}

	// Solve (L_T)x = y
	for (auto i = dim - 1; i < 0; i--)
	{
		accumulator = 0.0;
		for (auto j = i + 1; j < dim; j++)
		{
			accumulator += L[j * dim + i] * xs[time_point * dim + j];
		}
		
		xs[time_point * dim + i] = (y[i] - accumulator) / L[i * dim + i];
	}
}

__global__ void bTimepointGeneratorKernel(double* A, double* Bs, double* Y, double* J, double* E, double* F, double* temp, const int dim, const int time_slice) {
	const auto time_point = blockIdx.x * blockDim.x + threadIdx.x;
	const auto B = &Bs[time_point * dim];
	
	// Calculate J - YE
	for (auto i = 0; i < dim; i++)
	{
		temp[i] = J[i] - Y[i] * E[i] * cos(F[i] * time_point / time_slice);
	}

	// Multiply A by output
	for (auto i = 0; i < dim; i++)
	{
		auto sum = 0.0;
		for (auto j = 0; j < dim; j++)
		{
			sum += A[i * dim + j] * temp[j];
		}

		// Store in B
		B[i] = sum;
	}
}

// Perform Cholesky decomposition
void choleskyDecomposition(std::vector<double> A, std::vector<double> L, const int dim) {
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

double* computeCoefficientMatrix(cuda_mem::grid<double> A, std::vector<double> Y, const int dim)
{
	auto temp = new double[dim * dim];

	// Multiply A by Y
	for (auto i = 0; i < dim; i++)
	{
		for (auto j = 0; j < dim; j++)
		{
			temp[i * dim + j] = A[i][j] * Y[j];
		}
	}

	auto output = new double[dim * dim];

	// Multiply Y by transpose of A
	for (auto i = 0; i < dim; i++)
	{
		for (auto j = 0; j < dim; j++)
		{
			auto sum = 0.0;
			for (auto k = 0; k < dim; k++)
			{
				sum += A[i][j] * temp[i * dim + j];
			}

			output[i * dim + j] = sum;
		}
	}

	return output;
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
    dbg("Successfully parsed file: ", file_path);
    
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
	auto num_timepoints = 10 * static_cast<int>(*std::max_element(data.F.data(), data.F.data() + data.F.size())); // Set num_timepoints as 10*MAX(F)
	auto A = data.A;
	std::vector<double> L(data.n * data.n, 0.0);
	std::vector<double> Bs(data.timepoints * data.n, 10.0);
	std::vector<double> Xs(data.timepoints * data.n, 0.0);

	double* dev_A = 0;
	double* dev_L = 0;
	double* dev_Bs = 0;
	double* dev_Xs = 0;
	double* dev_Y = 0;

	// Get coefficient matrix
	auto A = computeCoefficientMatrix(*A.data(), data.Y, data.n);
	
	// Perform Cholesky decomposition
	choleskyDecomposition(*A.data(), L, data.n);

	// Print timepoints
	printf("L: \n");
	for (auto i = 0; i < data.n; i++)
	{
		
		for (auto j = 0; j < data.n; j++)
		{
			printf("%lf ", L[i * data.n + j]);
		}
		printf("\n");
	}

	// Choose which GPU to run on, change this on a multi-GPU system.
	auto cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for image and new image.
	cudaStatus = cudaMalloc((void**)& dev_A, data.n * data.n * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc for A array failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)& dev_L, data.n * data.n * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc for L array failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)& dev_Bs, num_timepoints * data.n * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc for Bs array failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)& dev_Xs, num_timepoints * data.n * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc for Xs array failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)& dev_Y, data.n * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc for Y array failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_A, A.data(), data.n * data.n * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy from A to dev_A failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_L, L.data(), data.n * data.n * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy from L to dev_L failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_Bs, Bs.data(), num_timepoints * data.n * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy from Bs to dev_Bs failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_Xs, Xs.data(), num_timepoints * data.n * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy from Xs to dev_Xs failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_Y, data.Y.data(), data.n * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy from Y to dev_Y failed!");
		goto Error;
	}

	// Perform solver for each timepoint
	choleskyTimepointSolverKernel << <1, num_timepoints >> > (dev_A, dev_L, dev_Bs, dev_Y, dev_Xs, data.n);

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
	cudaStatus = cudaMemcpy(Xs.data(), dev_Xs, num_timepoints * data.n * sizeof(double), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy from dev_Xs to Xs failed!");
		goto Error;
	}

	// Print timepoints
	for (auto i = 0; i < num_timepoints; i++)
	{
		printf("Timepoint %d", i + 1);
		for (auto j = 0; j < data.n; j++)
		{
			printf("%lf ", Xs[i * data.n + j]);
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