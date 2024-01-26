#pragma once
#include <stdint.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <thrust/device_vector.h>
class bitBoard
{
	unsigned long long pawns,
		rooks,
		knights,
		bishops,
		queens,
		kings;

	bool amBlack = false;
	
public:
	__device__ __host__ bool isBlack();

	/// <summary>
	/// Initialize a bitBoard with the default starting position using constants that are equivalent to the starting positions in binary
	/// </summary>
	/// <returns></returns>
	__device__ __host__ bitBoard();

	__host__ void printBoard();

	/// <summary>
	/// Flips the board vertically to represent the black side using stdlib.h's _byteswap_uint64. Can be used to flip the board back to the white side.
	/// </summary>
	/// <returns></returns>
	__device__ __host__ void flipSide();

	/// <summary>
	/// Generates all possible moves for the current board state. Returns a vector of bitBoards that represent all possible moves. This can be at most 218 moves.
	/// </summary>
	/// <param name="opponentBoard">The opponent's board state: Useful for pawn diagonal capture, eliminating obstructed bishop, rook and queen moves etc. </param>
	__device__ thrust::device_vector<bitBoard> generateAllStates(bitBoard opponentBoard) {
		thrust::device_vector<bitBoard> allStates;
		//pawn moves
		bool amBlackTemp = amBlack;	
		if (amBlack) {
			flipSide();
		}

		//PAWN MOVES

		//ROOK MOVES

		//KNIGHT MOVES

		//BISHOP MOVES

		//QUEEN MOVES

		//KING MOVES


		//Check Legality









		if (amBlackTemp) {
			flipSide();
		}
	}
};

