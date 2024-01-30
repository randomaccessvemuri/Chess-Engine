#include "bitBoard.cuh"

/// <summary>
/// Flips the board vertically to represent the black side using stdlib.h's _byteswap_uint64. Can be used to flip the board back to the white side.
/// </summary>
/// <returns></returns>

__device__ __host__ bool bitBoard::isBlack() {
	return amBlack;
}

__device__ __host__ bitBoard::bitBoard() {
	pawns = 65280; //Second row
	rooks = 129; //Corners of the first row
	knights = 66; //Next to the corners
	bishops = 36; //Next to the knights
	queens = 8; //Next to the bishops
	kings = 16; //In the middle of the first row
}

__device__ __host__ void bitBoard::printBoard() {
	unsigned long long mask = 1;
	for (int i = 0; i < 64; i++) {
		if (i % 8 == 0) {
			printf("\n");
		}
		if (pawns & mask) {
			printf("P ");
		}
		else if (rooks & mask) {
			printf("R ");
		}
		else if (knights & mask) {
			printf("N ");
		}
		else if (bishops & mask) {
			printf("B ");
		}
		else if (queens & mask) {
			printf("Q ");
		}
		else if (kings & mask) {
			printf("K ");
		}
		else {
			printf("0 ");
		}
		mask = mask << 1;
	}
	printf("\n");
}

__device__ __host__ void bitBoard::flipSide() {
	//byteswap reverses the order of the bits in the long long 
	auto byteswapper = [](unsigned long long x) {
		//Offsetting
		x = ((x >> 32) | (x << 32));

		x = ((x & 0xFFFF0000FFFF0000) >> 16) | ((x & 0x0000FFFF0000FFFF) << 16); // Swap the 16-bit halves
		x = ((x & 0xFF00FF00FF00FF00) >> 8) | ((x & 0x00FF00FF00FF00FF) << 8); // Swap the 8-bit halves
		return x;
		};
	pawns = byteswapper(pawns);
	rooks = byteswapper(rooks);
	knights = byteswapper(knights);
	bishops = byteswapper(bishops);
	queens = byteswapper(queens);
	kings = byteswapper(kings);
	amBlack = !amBlack;

}
