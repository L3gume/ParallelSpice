#define _USE_MATH_DEFINES

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <math.h>
#include <vector>
#include <string>
#include <iostream>
#include "JsonParser.h"
#include "dbg.h"
#include "cuda_utils.h"

cudaError_t solve(json::parse_result& data);
std::vector<double> choleskyDecomposition(std::vector<double> S, const int n);
std::vector<double> computeCoefficientMatrix(std::vector<std::vector<double>> A, std::vector<double> Y, const int n, const int m);

// Returns X_is where Xs are the node voltages associated with different timepoints
__global__ void choleskyTimepointSolverKernel(double* L, double* Bs, double* temp_solver, double* Xs, const int n) {
	const auto time_point = blockIdx.x * blockDim.x + threadIdx.x;
	const auto b = &Bs[time_point * n];
	auto accumulator = 0.0;

	// Solve L*temp_solver = b
	for (auto i = 0; i < n; i++)
	{
		accumulator = 0.0;

		for (auto j = 0; j < i; j++)
		{
			accumulator += L[i * n + j] * temp_solver[j];
		}

		temp_solver[i] = (b[i] - accumulator) / L[i * n + i];
	}

	// Solve (L_T)x = temp_solver
	const auto X = &Xs[time_point * n];
	for (auto i = n - 1; i >= 0; i--)
	{
		accumulator = 0.0;
		for (auto j = i + 1; j < n; j++)
		{
			accumulator += L[j * n + i] * X[j];
		}
		
		X[i] = (temp_solver[i] - accumulator) / L[i * n + i];
	}
}

// Returns a set of Bs where B_i is the forcing function at a timepoint
__global__ void bTimepointGeneratorKernel(double* A, double* Bs, double* Y, double* J, double* E, double* F, double* temp_generator, const int n, const int m, const int T_max, const int num_samples) {
	const auto time_point = blockIdx.x * blockDim.x + threadIdx.x;
	const auto B = &Bs[time_point * n];
	
	// Calculate J - YE
	for (auto i = 0; i < m; i++)
	{
		temp_generator[i] = J[i] - Y[i] * E[i] * cos(2 * M_PI * F[i] * time_point * T_max / num_samples);
	}

	// Multiply A by output
	for (auto i = 0; i < n; i++)
	{
		auto sum = 0.0;
		for (auto j = 0; j < m; j++)
		{
			sum += A[i * n + j] * temp_generator[j];
		}

		// Store in B
		B[i] = sum;
	}
}

// Perform Cholesky decomposition
std::vector<double> choleskyDecomposition(std::vector<double> S, const int n) {
	std::vector<double> L(n * n, 0.0);
	auto accumulator = 0.0;

	for (auto i = 0; i < n; i++)
	{
		for (auto j = 0; j < (i + 1); j++)
		{
			accumulator = 0.0;
			for (auto k = 0; k < j; k++)
			{
				accumulator += L[i * n + k] * L[j * n + k];
			}

			if (i == j)
			{
				L[i * n + j] = sqrt(S[i * n + i] - accumulator);
			}
			else
			{
				L[i * n + j] = (1.0 / L[j * n + j]) * (S[i * n + j] - accumulator);
			}
		}
	}

	return L;
}

std::vector<double> computeCoefficientMatrix(std::vector<std::vector<double>> A, std::vector<double> Y, const int n, const int m)
{
	std::vector<double> temp(m * n, 0.0);

	// Multiply A by Y
	for (auto i = 0; i < m; i++)
	{
		for (auto j = 0; j < n; j++)
		{
			temp[i * m + j] = A[i][j] * Y[j];
		}
	}

	std::vector<double> output(n * n, 0.0);

	// Multiply Y by transpose of A
	for (auto i = 0; i < m; i++)
	{
		for (auto j = 0; j < n; j++)
		{
			auto sum = 0.0;
			for (auto k = 0; k < m; k++)
			{
				sum += A[k][i] * temp[k * m + j];
			}

			output[i * n + j] = sum;
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
	const auto T_max = static_cast<int>(*std::max_element(data.F.data(), data.F.data() + data.F.size()));
	auto num_timepoints = 2 * std::max(1, T_max); // Set num_timepoints as 10*MAX(F)
	
	std::vector<double> Bs(num_timepoints * data.n, 0.0);
	std::vector<double> Xs(num_timepoints * data.n, 0.0);
	std::vector<double> temp_solver(data.n, 0.0);
	std::vector<double> temp_generator(data.n, 0.0);

	/*auto grid_device = cuda_mem::grid_to_cuda_2d(data.A);
	auto g = cuda_mem::cuda_2d_to_grid(grid_device);

	auto vec_device = cuda_mem::make_cuda_unique(data.E);
	auto vec = cuda_mem::cuda_unique_to_vector(vec_device);*/

	double* dev_A = 0;
	double* dev_S = 0;
	double* dev_L = 0;
	double* dev_Bs = 0;
	double* dev_Xs = 0;
	double* dev_Y = 0;
	double* dev_J = 0;
	double* dev_E = 0;
	double* dev_F = 0;
	double* dev_temp_solver = 0;
	double* dev_temp_generator = 0;

	// Get coefficient matrix
	auto S = computeCoefficientMatrix(data.A, data.Y, data.n, data.m);
		
	// Perform Cholesky decomposition
	auto L = choleskyDecomposition(S, data.n);

	// Choose which GPU to run on, change this on a multi-GPU system.
	auto cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for image and new image.
	cudaStatus = cudaMalloc((void**)& dev_A, data.m * data.n * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc for A array failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)& dev_S, data.n * data.n * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc for S array failed!");
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

	cudaStatus = cudaMalloc((void**)& dev_Y, data.m * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc for Y array failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)& dev_J, data.m * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc for J array failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)& dev_E, data.m * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc for E array failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)& dev_F, data.m * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc for F array failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)& dev_temp_solver, data.n * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc for temp solver array failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)& dev_temp_generator, data.n * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc for temp solver array failed!");
		goto Error;
	}

	auto a = new double[data.m * data.n];
	for (auto i = 0; i < data.m; i++)
	{
		for (auto j = 0; j < data.n; j++)
		{
			a[i * data.m + j] = data.A[i][j];
		}
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_A, a, data.m * data.n * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy from A to dev_A failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_S, S.data(), data.n * data.n * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy from S to dev_S failed!");
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

	cudaStatus = cudaMemcpy(dev_Y, data.Y.data(), data.m * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy from Y to dev_Y failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_J, data.J.data(), data.m * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy from J to dev_J failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_E, data.E.data(), data.m * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy from E to dev_E failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_F, data.F.data(), data.m * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy from F to dev_F failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_temp_solver, temp_solver.data(), data.n * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy from temp_solver to dev_temp_solver failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_temp_generator, temp_generator.data(), data.n * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy from temp_generator to dev_temp_generator failed!");
		goto Error;
	}

	int num_blocks = 0;
	int num_threads = 0;
	cuda_utils::divide_threads_into_blocks(num_timepoints, num_blocks, num_threads);

	// Generate Bs
	bTimepointGeneratorKernel << <num_blocks, num_threads >> > (dev_A, dev_Bs, dev_Y, dev_J, dev_E, dev_F, dev_temp_generator, data.n, data.m, T_max, num_timepoints);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "bTimepointGeneratorKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching bTimepointGeneratorKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(Bs.data(), dev_Bs, num_timepoints * data.n * sizeof(double), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy from dev_Bs to Bs failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_Bs, Bs.data(), num_timepoints * data.n * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy from Bs to dev_Bs failed!");
		goto Error;
	}

	// Perform solver for each timepoint
	choleskyTimepointSolverKernel<<<num_blocks, num_threads >>>(dev_L, dev_Bs, dev_temp_solver, dev_Xs, data.n);

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
		printf("Timepoint %d\n", i + 1);
		for (auto j = 0; j < data.n; j++)
		{
			printf("%lf ", Xs[i * data.n + j]);
		}
		printf("\n");
	}

	auto temp = new double[data.n];
	for (auto i = 0; i < data.n; i++)
	{
		temp[i] = 0;
		for (auto j = 0; j < data.n; j++)
		{
			temp[i] += S[i * data.n + j] * Xs[j];
		}
	}

	// Print timepoints
	printf("Sanity: ");
	for (auto i = 0; i < data.n; i++)
	{
		printf("%lf ", temp[i]);
	}
			

Error:
	cudaFree(dev_A);
	cudaFree(dev_S);
	cudaFree(dev_L);
	cudaFree(dev_Bs);
	cudaFree(dev_Xs);
	cudaFree(dev_Y);
	cudaFree(dev_J);
	cudaFree(dev_E);
	cudaFree(dev_F);
	cudaFree(dev_temp_solver);
	cudaFree(dev_temp_generator);

	return cudaStatus;
}