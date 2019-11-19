#pragma once

#include "cuda_runtime.h"
#include <cassert>
#include <utility>
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

		T operator->() const noexcept;
		T operator*() const noexcept;
		T* get() const noexcept;
		void release() noexcept;

	private:
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
        std::copy(&ptr[0], &ptr[0] + size, m_ptr);
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
			delete[] m_ptr;
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
    
    // 2D array implementation
    // Typedef for grid
    template <typename T>
    using grid = std::vector<std::vector<T>>;
    
    template <typename T>
    grid<T> make_grid(const int n, const T val) {
        grid<T> g;
        g.assign(n, std::vector<T>());
        for (auto& sub : g) {
            sub.assign(n, val);
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
    grid<T> cuda_unique_to_grid(const int n, const T* ptr) {
        auto ret = grid<T>{};
        ret.reserve(n);
        for (auto i = 0; i < n; ++i) {
            ret[i] = std::vector<T>(&ptr[n * i], &ptr[n * i] + n);
        }
        return ret;
    }
    
}