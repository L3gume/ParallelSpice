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

constexpr auto BASE_RATE = 20.0;

cudaError_t solve(json::parse_result& data, std::string out_path);
std::vector<double> choleskyDecomposition(std::vector<double> S, const int m);
std::vector<double> computeCoefficientMatrix(std::vector<std::vector<double>> A, std::vector<double> Y, const int n, const int m);

// Genereates a set of Bs where B_i is the forcing function at a timepoint.
// Returns X_is where Xs are node voltages associated with different timepoints based on the generated B_is.
__global__ void bTimepointGeneratorAndSolverKernel(double* A, double* L, double* Bs, double* Xs, double* Y, double* J, double* E, double* VF, double* IF, double* temp_generator, double* temp_solver, const int n, const int m, const double delta_T) {
	const auto time_point = blockIdx.x * blockDim.x + threadIdx.x;
	const auto B = &Bs[time_point * m];
	const auto temp_G = &temp_generator[time_point * n];
	const auto temp_S = &temp_solver[time_point * m];
	
	// Calculate J - YE taking VF and IF into account
	for (auto i = 0; i < n; i++)
	{
		temp_G[i] = (J[i] * cos(2 * M_PI * IF[i] * time_point * delta_T)) - Y[i] * E[i] * cos(2 * M_PI * VF[i] * time_point * delta_T);
	}

	// Multiply A by output
	for (auto i = 0; i < m; i++)
	{
		auto sum = 0.0;
		for (auto j = 0; j < n; j++)
		{
			sum += A[i * n + j] * temp_G[j];
		}

		// Store in B
		B[i] = sum;
	}

	auto accumulator = 0.0;

	// Solve L*temp_solver = b
	for (auto i = 0; i < m; i++)
	{
		accumulator = 0.0;

		for (auto j = 0; j < i; j++)
		{
			accumulator += L[i * m + j] * temp_S[j];
		}

		temp_S[i] = (B[i] - accumulator) / L[i * m + i];
	}

	// Solve (L_T)x = temp_solver
	const auto X = &Xs[time_point * m];
	for (auto i = m - 1; i >= 0; i--)
	{
		accumulator = 0.0;
		for (auto j = i + 1; j < m; j++)
		{
			accumulator += L[j * m + i] * X[j];
		}

		X[i] = (temp_S[i] - accumulator) / L[i * m + i];
	}
}

// Perform Cholesky decomposition
std::vector<double> choleskyDecomposition(std::vector<double> S, const int m) {
	std::vector<double> L(m * m, 0.0);
	auto accumulator = 0.0;

	for (auto i = 0; i < m; i++)
	{
		for (auto j = 0; j < (i + 1); j++)
		{
			accumulator = 0.0;
			for (auto k = 0; k < j; k++)
			{
				accumulator += L[i * m + k] * L[j * m + k];
			}

			if (i == j)
			{
				L[i * m + j] = sqrt(S[i * m + i] - accumulator);
			}
			else
			{
				L[i * m + j] = (1.0 / L[j * m + j]) * (S[i * m + j] - accumulator);
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
			temp[i * n + j] = A[i][j] * Y[j];
		}
	}

	std::vector<double> output(m * m, 0.0);

	// Multiply AY by transpose of A
	for (auto i = 0; i < m; i++)
	{
		for (auto j = 0; j < m; j++)
		{
			auto sum = 0.0;
			for (auto k = 0; k < n; k++)
			{
				sum += A[i][k] * temp[j * n + k];
			}

			output[i * m + j] = sum;
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
	const auto F_V_max = *std::max_element(data.VF.data(), data.VF.data() + data.VF.size());
	const auto F_I_max = *std::max_element(data.IF.data(), data.IF.data() + data.IF.size());
	const auto F_max = std::max(F_V_max, F_I_max);

	auto F_min = F_max;
	for (auto i = 0; i < data.VF.size(); i++)
	{
		if (data.VF[i] != 0.0 && data.VF[i] < F_min)
			F_min = data.VF[i];

		if (data.IF[i] != 0.0 && data.IF[i] < F_min)
			F_min = data.IF[i];
	}

	auto T_max = 1 / F_min;
	auto T_min = 1 / F_max;

	auto delta_T = T_min / BASE_RATE;
	auto num_timepoints = static_cast<int>((T_max / T_min) / delta_T);
	
	std::vector<double> Bs(num_timepoints * data.m, 0.0);
	std::vector<double> Xs(num_timepoints * data.m, 0.0);
	std::vector<double> temp_solver(num_timepoints * data.m, 0.0);
	std::vector<double> temp_generator(num_timepoints * data.n, 0.0);

	// Get coefficient matrix
	auto S = computeCoefficientMatrix(data.A, data.Y, data.n, data.m);
		
	// Perform Cholesky decomposition
	auto L = choleskyDecomposition(S, data.m);

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
	bTimepointGeneratorAndSolverKernel << <num_blocks, num_threads >> > (device_a.get(), device_l.get(), device_bs.get(), device_xs.get(), device_y.get(), device_j.get(), device_e.get(), device_vf.get(), device_vi.get(), device_temp_g.get(), device_temp_s.get(), data.n, data.m, delta_T);

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

	// Get solution for X from device
	auto X_solution = cuda_mem::cuda_unique_to_vector(device_xs);
	
	// PERFORM SANITY CHECK
	// Get Sx for verification
	std::vector<double> S_times_X(num_timepoints * data.m, 0.0);
	for (auto t = 0; t < num_timepoints; t++)
	{
		for (auto i = 0; i < data.m; i++)
		{
			for (auto j = 0; j < data.m; j++)
			{
				S_times_X[t * data.m + i] += S[i * data.m + j] * X_solution[t * data.m + j];
			}
		}
	}
	Bs = cuda_mem::cuda_unique_to_vector(device_bs);
	// Verify that Sx = b and print error
	auto error = 0.0;
	for (auto i = 0; i < num_timepoints; i++)
	{
		for (auto j = 0; j < data.m; j++)
		{
			error += S_times_X[i * data.m + j] - Bs[i * data.m + j];
		}
	}
	printf("Sanity: %lf", error);

	// Covnert Xs to grid for output
	auto out_grid = cuda_mem::grid<double>();
	for (auto i = 0; i < num_timepoints; i++)
	{
		out_grid.emplace_back(std::vector<double>(&X_solution[i * data.m], &X_solution[i * data.m] + data.m));
	}

	// Write to json file
	auto writer = json::JsonWriter(out_path);
	if (!writer.write(delta_T, out_grid)) {
		return cudaError(1);
	}

	return cudaSuccess;
}