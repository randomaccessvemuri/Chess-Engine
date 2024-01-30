#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include "bitBoard.cuh"


int main() {
	bitBoard board;
	board.printBoard();
	board.knights <<= 18;
	std::cout << board.knights;
	board.printBoard();



	return 0;
}