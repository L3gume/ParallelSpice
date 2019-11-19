#pragma once
#include "cuda_runtime.h"
#include <vector>
#include <algorithm>
#include <fstream>
#include <cassert>

namespace cuda_utils {

static constexpr auto thread_limit = 1024;

// Compute the thread/block spread, in case the number of threads is higher than the limit allowed (1024).
// As long as we are higher than the limit, divide the number of threads per block by two and multiply
// the number of blocks by two.
// returns (n_blocks, n_threads) pair
inline void divide_threads_into_blocks(const int n_threads, int& blocks, int& threads) {
    blocks = 1;
    threads = n_threads;
    while (n_threads > thread_limit) {
        blocks *= 2;
        threads /= 2;
    }
}

// Simple check to make sure the file exists
inline bool file_exists(const char *file_name) {
    const std::ifstream in_file(file_name);
    return in_file.good();
}

template <typename T>
__device__ uint8_t clamp_val_to_uint8(T val) {
    const auto min = T{0};
    const auto max = T{255};
    return static_cast<uint8_t>(val > min ? (val < max ? val : max) : min);
}

inline std::string make_output_file(const std::string& in_file, const std::string& suffix, const std::string& extension = "png") {
    const auto last_index = in_file.find_last_of(".");
    const auto raw_name = in_file.substr(0, last_index);
    return raw_name + suffix + "." + extension;
}

}
