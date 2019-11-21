/*
 * cuda_mem.h - some wrappers around CUDA device memory. Meant to make working with device memory easier
 * author: Justin Tremblay, 2019
 */
#pragma once

#include "cuda_runtime.h"
#include <cassert>
#include <iostream>
#include <vector>

/*
 * CUDA smart pointers
 *
 * contains implementations for unique pointer wrapping around the
 * CUDA allocation and free calls. (shared_ptr didn't seem necessary here)
 *
 * also contains cuda_device_ptr, which is meant to copy back from the device
 * to host memory
 */

namespace cuda_mem {

// Can only be moved, rendering the moved pointer unusable 
template<typename T>
class cuda_unique_ptr {
public:
    cuda_unique_ptr(size_t size, const T* ptr);

    cuda_unique_ptr(const cuda_unique_ptr&) = delete;
    cuda_unique_ptr operator=(const cuda_unique_ptr&) = delete;

    // Transfer ownership of the memory, 
    cuda_unique_ptr(cuda_unique_ptr<T>&&) noexcept;
    cuda_unique_ptr operator=(cuda_unique_ptr<T>&&) noexcept;

    ~cuda_unique_ptr();

    T& operator[](int index);
    T operator->() const noexcept;
    T operator*() const noexcept;
    T* get() const noexcept;
    void release() noexcept;

protected:
    T* m_ptr;
    cudaError_t m_last_error;
    size_t m_size = 0;
};

template <typename T>
cuda_unique_ptr<T>::cuda_unique_ptr(const size_t size, const T* ptr) {
    m_size = size;
    m_ptr = nullptr;
    m_last_error = cudaMallocManaged(reinterpret_cast<void**>(&m_ptr), size * sizeof(T));
    assert(m_last_error == cudaSuccess);

    if (ptr != nullptr) {
        std::copy(&ptr[0], &ptr[0] + size, m_ptr);                        
    }
}

template <typename T>
cuda_unique_ptr<T>::cuda_unique_ptr(cuda_unique_ptr&& rhs) noexcept {
    m_ptr = nullptr;
    m_size = rhs.m_size;
    m_last_error = cudaMallocManaged(reinterpret_cast<void**>(&m_ptr), m_size * sizeof(T));
    assert(m_last_error == cudaSuccess);

    std::copy(&rhs.m_ptr[0], &rhs.m_ptr[0] + m_size, m_ptr);
    rhs.release();
}

template <typename T>
cuda_unique_ptr<T> cuda_unique_ptr<T>::operator=(cuda_unique_ptr&& rhs) noexcept {
    return cuda_unique_ptr(rhs);
}

template <typename T>
cuda_unique_ptr<T>::~cuda_unique_ptr() {
    if (m_ptr != nullptr) cudaFree(m_ptr);
}

template <typename T>
T& cuda_unique_ptr<T>::operator[](const int index) {
    // warning: doesn't check bounds
    return m_ptr[index];
}

template <typename T>
T cuda_unique_ptr<T>::operator->() const noexcept {
    return *m_ptr;
}

template <typename T>
T cuda_unique_ptr<T>::operator*() const noexcept {
    return *m_ptr;
}

template <typename T>
T* cuda_unique_ptr<T>::get() const noexcept {
    return m_ptr;
}

template <typename T>
void cuda_unique_ptr<T>::release() noexcept {
    if (m_ptr != nullptr) {
        cudaFree(m_ptr);
        m_ptr = nullptr;
    }
}

template <typename T>
static cuda_unique_ptr<T> make_cuda_unique(const size_t size, const T* ptr) {
    return cuda_unique_ptr<T>(size, ptr);
}

template<typename T>
static cuda_unique_ptr<T> make_cuda_unique(const std::vector<T>& vec) {
    return cuda_unique_ptr<T>(vec.size(), vec.data());
}

template<typename T>
class cuda_device_ptr {
public:
	cuda_device_ptr(size_t size, const T* ptr);
	~cuda_device_ptr();

	T operator->() const noexcept;
	T operator*()const noexcept;

	T* get() const noexcept;
	void release() noexcept;

	T* begin() const noexcept;
	T* end() const noexcept;

private:
	T* m_ptr = nullptr;
	size_t m_size = 0;
};

template<typename T>
cuda_device_ptr<T>::cuda_device_ptr(const size_t size, const T* ptr) {
    m_size = size;
    m_ptr = new T[size];
    cudaMemcpy(m_ptr, ptr, size * sizeof(T), cudaMemcpyDeviceToHost);
}

template<typename T>
cuda_device_ptr<T>::~cuda_device_ptr() {
	release();
}

template<typename T>
T cuda_device_ptr<T>::operator->() const noexcept {
	return T();
}

template<typename T>
T cuda_device_ptr<T>::operator*() const noexcept {
	return T();
}

template<typename T>
T* cuda_device_ptr<T>::get() const noexcept {
	return m_ptr;
}

template<typename T>
void cuda_device_ptr<T>::release() noexcept {
	if (m_ptr != nullptr) {
        cudaFree(m_ptr);
		m_ptr = nullptr;
	}
}

template<typename T>
T* cuda_device_ptr<T>::begin() const noexcept {
	return m_ptr;
}

template<typename T>
T* cuda_device_ptr<T>::end() const noexcept {
	return m_ptr + m_size;
}

template<typename T>
static cuda_device_ptr<T> make_cuda_device(const size_t size, const T* ptr) {
   return cuda_device_ptr<T>(size, ptr);
}

// cuda_unique_2d
// inherits from cuda_unique_ptr
// behaves the same except it has an extra member function which allows 2d indexing

template <typename T>
class cuda_unique_2d : public cuda_unique_ptr<T> {

public:
    cuda_unique_2d(size_t size, size_t width, size_t height, const T* ptr);
    
    cuda_unique_2d(const cuda_unique_2d&) = delete;
    cuda_unique_2d operator=(const cuda_unique_2d&) = delete;
    
    cuda_unique_2d(cuda_unique_2d<T>&& rhs) noexcept;
    cuda_unique_2d operator=(cuda_unique_2d<T>&& rhs) noexcept;

    T& at(int x, int y);
private:
    size_t m_width;
    size_t m_height;
};

template <typename T>
cuda_unique_2d<T>::cuda_unique_2d(const size_t size, const size_t width, const size_t height, const T* ptr)
    : cuda_unique_ptr<T>(size, ptr), m_width(width), m_height(height) {
}


template <typename T>
cuda_unique_2d<T>::cuda_unique_2d(cuda_unique_2d<T>&& rhs) noexcept: cuda_unique_ptr<T>(rhs) {
    m_width = rhs.m_width;
    m_height = rhs.m_height;
}

template <typename T>
cuda_unique_2d<T> cuda_unique_2d<T>::operator=(cuda_unique_2d<T>&& rhs) noexcept {
    return cuda_unique_2d<T>(rhs);
}

template <typename T>
T& cuda_unique_2d<T>::at(const int x, const int y) {
    return m_ptr[y * m_width + x];
}

// make cuda 2d
template<typename T>
static cuda_unique_2d<T> make_cuda_2d(const std::vector<T>& vec, const int width, const int height) {
    return cuda_unique_2d<T>(vec.size(), width, height, vec.data());
}

// Cuda unique to vector, cuda device may actually not be necessary anymore
template<typename T>
static std::vector<T> cuda_unique_to_vector(const size_t size, const cuda_unique_ptr<T>& ptr) {
    const auto temp = make_cuda_device(size, ptr.get());
    return std::vector<T>(temp.begin(), temp.end());
}

// 2D array implementation
// Typedef for grid
template <typename T>
using grid = std::vector<std::vector<T>>;

template <typename T>
grid<T> make_grid(const int width, const int height, const T val) {
    grid<T> g;
    //g.assign(height, std::vector<T>());
    g.resize(height);
    for (auto& sub : g) {
        sub.assign(width, val);
    }
    return g;
}

template <typename T>
cuda_unique_ptr<T> grid_to_cuda_unique(const grid<T>& grid) {
    auto total_size = 0;
    for (const auto& sub : grid) {
        total_size += sub.size();
    }
    std::vector<T> temp;
    temp.reserve(total_size);
    for (const auto& sub : grid) {
        temp.insert(temp.end(), sub.begin(), sub.end());
    }
    return make_cuda_unique<T>(total_size, temp.data());
}

template <typename T>
cuda_unique_2d<T> grid_to_cuda_2d(const grid<T>& grid) {
    auto total_size = 0;
    auto height = grid.size();
    auto width = grid[0].size();
    for (const auto& sub : grid) {
        total_size += sub.size();
    }
    std::vector<T> temp;
    temp.reserve(total_size);
    for (const auto& sub : grid) {
        temp.insert(temp.end(), sub.begin(), sub.end());
    }
    return make_cuda_2d<T>(temp, width, height);
}

template <typename T>
grid<T> cuda_2d_to_grid(const int width, const int height, const cuda_unique_2d<T>& cuda_ptr) {
    auto ret = grid<T>{};
    ret.resize(height);
    auto tmp = make_cuda_device(width * height, cuda_ptr.get());
    auto ptr = tmp.get();
    for (auto i = 0; i < width; ++i) {
        std::copy(&ptr[width * i], &ptr[width * i] + width, std::back_inserter(ret[i]));
    }
    return ret;
}
    
}