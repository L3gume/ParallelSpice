#define _USE_MATH_DEFINES

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <math.h>
#include <vector>
#include <string>
#include <iostream>
#include "JsonParser.h"
#include "JsonWriter.h"
#include "dbg.h"
#include "cuda_utils.h"

cudaError_t solve(json::parse_result& data, std::string out_path);
std::vector<double> choleskyDecomposition(std::vector<double> S, const int n);
std::vector<double> computeCoefficientMatrix(std::vector<std::vector<double>> A, std::vector<double> Y, const int n, const int m);

// Genereates a set of Bs where B_i is the forcing function at a timepoint.
// Returns X_is where Xs are node voltages associated with different timepoints based on the generated B_is.
__global__ void bTimepointGeneratorAndSolverKernel(double* A, double* L, double* Bs, double* Xs, double* Y, double* J, double* E, double* VF, double* IF, double* temp_generator, double* temp_solver, const int n, const int m, const int T_max, const int num_samples) {
	const auto time_point = blockIdx.x * blockDim.x + threadIdx.x;
	const auto B = &Bs[time_point * n];
	
	// Calculate J - YE
	for (auto i = 0; i < m; i++)
	{
		temp_generator[i] = J[i] - Y[i] * E[i] * cos(2 * M_PI * VF[i] * time_point * T_max / num_samples);
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

	auto accumulator = 0.0;

	// Solve L*temp_solver = b
	for (auto i = 0; i < n; i++)
	{
		accumulator = 0.0;

		for (auto j = 0; j < i; j++)
		{
			accumulator += L[i * n + j] * temp_solver[j];
		}

		temp_solver[i] = (B[i] - accumulator) / L[i * n + i];
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
        parallelSpice.exe [input json file path])" << '\n';
        exit(1);
    }
	
	// Parse json file with circuit configuration
    const auto in_file_path = std::string{argv[1]};
    auto result = json::parse_result{};
    auto parser = json::JsonParser(in_file_path);
    if (!parser.parse(result)) {
        exit(1);
    }
    dbg("Successfully parsed file: ", in_file_path);
	    
    // Solve.
    auto cudaStatus = solve(result, in_file_path);
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

cudaError_t solve(json::parse_result& data, std::string out_path) {
	const auto T_max = static_cast<int>(*std::max_element(data.VF.data(), data.VF.data() + data.VF.size()));
	const auto num_timepoints = 2 * std::max(1, T_max); // Set num_timepoints as 2*MAX(F)
	const auto time_step = 0;
	
	std::vector<double> Bs(num_timepoints * data.n, 0.0);
	std::vector<double> Xs(num_timepoints * data.n, 0.0);
	std::vector<double> temp_solver(data.n, 0.0);
	std::vector<double> temp_generator(data.n, 0.0);

	// Get coefficient matrix
	auto S = computeCoefficientMatrix(data.A, data.Y, data.n, data.m);
		
	// Perform Cholesky decomposition
	auto L = choleskyDecomposition(S, data.n);

	const auto& device_a = cuda_mem::grid_to_cuda_2d(data.A);
	const auto& device_l = cuda_mem::vec_to_cuda_unique(L);
	const auto& device_bs = cuda_mem::vec_to_cuda_unique(Bs);
	const auto& device_xs = cuda_mem::vec_to_cuda_unique(Xs);
	const auto& device_y = cuda_mem::vec_to_cuda_unique(data.Y);
	const auto& device_j = cuda_mem::vec_to_cuda_unique(data.J);
	const auto& device_e = cuda_mem::vec_to_cuda_unique(data.E);
	const auto& device_vf = cuda_mem::vec_to_cuda_unique(data.VF);
	const auto& device_vi = cuda_mem::vec_to_cuda_unique(data.IF);
	const auto& device_temp_g = cuda_mem::vec_to_cuda_unique(temp_generator);
	const auto& device_temp_s = cuda_mem::vec_to_cuda_unique(temp_solver);

	// Choose which GPU to run on, change this on a multi-GPU system.
	auto cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		return cudaStatus;
	}

	int num_blocks = 0;
	int num_threads = 0;
	cuda_utils::divide_threads_into_blocks(num_timepoints, num_blocks, num_threads);

	// Generate Bs and sovle for Xs
	bTimepointGeneratorAndSolverKernel << <num_blocks, num_threads >> > (device_a.get(), device_l.get(), device_bs.get(), device_xs.get(), device_y.get(), device_j.get(), device_e.get(), device_vf.get(), device_vi.get(), device_temp_g.get(), device_temp_s.get(), data.n, data.m, T_max, num_timepoints);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "bTimepointGeneratorAndSolverKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		return cudaStatus;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching bTimepointGeneratorKernel!\n", cudaStatus);
		return cudaStatus;
	}

	auto solution = cuda_mem::cuda_unique_to_vector(device_xs);

	// Print timepoints
	for (auto i = 0; i < num_timepoints; i++)
	{
		printf("Timepoint %d\n", i + 1);
		for (auto j = 0; j < data.n; j++)
		{
			printf("%lf ", solution[i * data.n + j]);
		}
		printf("\n");
	}

	auto temp = new double[data.n];
	for (auto i = 0; i < data.n; i++)
	{
		temp[i] = 0;
		for (auto j = 0; j < data.n; j++)
		{
			temp[i] += S[i * data.n + j] * solution[j];
		}
	}

	// Print timepoints
	printf("Sanity: ");
	for (auto i = 0; i < data.n; i++)
	{
		printf("%lf ", temp[i]);
	}

	// Covnert Xs to grid for output
	auto out_grid = cuda_mem::grid<double>();
	for (auto i = 0; i < num_timepoints; i++)
	{
		out_grid.emplace_back(std::vector<double>(&solution[i * data.n], &solution[i * data.n] + data.n));
	}

	auto writer = json::JsonWriter(out_path);
	writer.write(time_step, out_grid);

	return cudaSuccess;
}