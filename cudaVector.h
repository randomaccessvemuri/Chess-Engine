#pragma once
/*
* This exists because the CUDA developers in their infinite wisdom haven't made a dynamically resizable vector class which can be used on the GPU. I assume it's because it's too expensive (computationally) to do so. But since I need vectors because there can be a indeterminate number of certain chess pieces due to promotions, I'll have to make my own!

*/
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <iostream>
#include <assert.h>



namespace internals {
	template <typename T>
	class cudaVector
	{
		T* data;
		unsigned long long capacity;
		unsigned long long size;

public:

	//Constructor: Dynamically allocates memory for the vector
	__host__ __device__ cudaVector(unsigned long long length) {
		data = new T[length];
		capacity = length;
		size = 0;
	}
	__host__ __device__ ~cudaVector() {
		delete[] data;
	}

	//Subscript operator: Get element at index
	__host__ __device__ T& operator[](unsigned long long index) {
		assert(index < size);
		return data[index];
	}

	//Push back: Add element to end of vector
	__host__ __device__ void push_back(T element) {
		if (capacity == size) {
			resize(capacity * 2);
		}

		data[size] = element;
		size++;	
	}

	//Pop back: Remove element from end of vector
	__host__ __device__ T pop_back() {
		if (size == 0) {
			return NULL;
		}
		else {
			T temp = data[size - 1];
			size--;
			return temp;
		}
	}

	//Get size of vector
	__host__ __device__ int getSize() {
		return size;
	}

	__host__ __device__ int getCapacity() {
		return capacity;
	}

	//TODO: There may be a more efficient way to do this. I'm not sure if this is the best way to do it.
	//*Resize vector: This is where we need to focus the most. Any slowdown here will be costly since anything related to this class will be using this function.
	__host__ __device__ void resize(unsigned long long newSize) {
		T* temp = new T[newSize];
		for (int i = 0; i < size; i++) {
			temp[i] = data[i];
		}
		delete[] data;
		data = temp;
		capacity = newSize;
	}


	__host__ __device__ void reduce() {
		T* temp = new T[size];
		for (int i = 0; i < size; i++) {
			temp[i] = data[i];
		}
		delete[] data;
		data = temp;
		capacity = size;
	}

	__host__ T* getPtr() {
		return data;
	}

	//NOTE: I actually don't know anything about how to do this. I'm just copying what I see on the internet from https://stackoverflow.com/questions/39811007/can-a-user-defined-class-have-custom-behavior-with-stdcout
	__host__ friend std::ostream& operator<<(std::ostream& os, const cudaVector<T>& vec) {
		for (int i = 0; i < vec.size; i++) {
			os << vec.data[i] << " ";
		}
		return os;
	}

	};

}


