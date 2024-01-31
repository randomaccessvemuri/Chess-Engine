#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include "bitBoard.cuh"

//Dummy run move generation on single GPU thread
__global__ void runMoveGen(bitBoard* board) {
	board->generateAllStates(*board);
}


int main() {
	bitBoard board;
	
	runMoveGen<<<1,1>>>(&board);
	
	


	return 0;
}