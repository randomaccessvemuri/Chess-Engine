/*
	Bitshifting to generate moves:
		Bitshift left 8 times to move the piece down a row w.r.t the printBoard() method
		Bitshift left or right once to move the piece left or right


	The flipSide thing is to not have to write the same code for black and white pieces. It's easier to just flip the board and then flip it back after generating the moves.

*/
#define DEBUG_MODE 

#pragma once
#include <stdint.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include "cudaVector.h"





/// <summary>
/// TODO: Documentation
/// </summary>
struct movesList {
	int pawnMoves;
	int rookMoves;
	int knightMoves;
	int bishopMoves;
	int queenMoves;
	int kingMoves;
	bool bShouldBeBlack;

	unsigned long long *allChanges;

};

class bitBoard
{
public:
	unsigned long long pawns,
		rooks,
		knights,
		bishops,
		queens,
		kings;

	bool amBlack = false;
	

	__device__ __host__ bool isBlack();

	/// <summary>
	/// Initialize a bitBoard with the default starting position using constants that are equivalent to the starting positions in binary
	/// </summary>
	/// <returns></returns>
	__device__ __host__ bitBoard();

	__device__ __host__ void printBoard();

	/// <summary>
	/// Flips the board vertically to represent the black side using stdlib.h's _byteswap_uint64. Can be used to flip the board back to the white side.
	/// </summary>
	/// <returns></returns>
	__device__ __host__ void flipSide();

	/// <summary>
	/// Generates all possible moves for the current board state. Returns a vector of bitBoards that represent all possible moves. This can be at most 218 moves.
	/// </summary>
	/// <TODO>
	/// - Eliminate branching as much as possible
	/// </TODO>
	/// <param name="opponentBoard">The opponent's board state: Useful for pawn diagonal capture, eliminating obstructed bishop, rook and queen moves etc. </param>
	__device__ movesList generateAllStates(bitBoard opponentBoard) {
		//The logic here is to reduce the amount of data stored in this and also that I can't really create a device_vector for bitBoard. So I'm going to only store the changes in the board state, which of the changes are for which piece and then generate the new board states as needed
		unsigned long long allChanges[218];
		int pawnChanges = 0;
		int rookChanges = 0;
		int knightChanges = 0;
		int bishopChanges = 0;
		int queenChanges = 0;
		int kingChanges = 0;		
		bool amBlackTemp = amBlack;		

		if (amBlack) {
			this->flipSide();


			//Flip the opponent's board as well
			opponentBoard.flipSide();
		}

		unsigned char currentBitIndex = 0; //reused for all move generations
		bool currentPawnBit = 0; 
		bool currentRookBit = 0; 
		bool currentKnightBit = 0;
		bool currentBishopBit = 0;
		bool currentQueenBit = 0;
		bool currentKingBit = 0;

		//PAWN MOVES (64 because each of the 8 pawns can make at most 8 possible moves)
		internals::cudaVector<unsigned long long> pawnMoves(64);

		//ROOK MOVES (30 because each of the 2 rooks can make at most 14 possible moves and 2 for castling)
		internals::cudaVector<unsigned long long> rookMoves(30);

		//KNIGHT MOVES (16 because each of the 2 knights can make at most 8 possible moves)
		internals::cudaVector<unsigned long long> knightMoves(16);

		//BISHOP MOVES (26 because each of the 2 bishops can make at most 13 possible moves)
		internals::cudaVector<unsigned long long> bishopMoves(26);


		while (currentBitIndex!=64) {
			
			//PAWN MOVES
			currentPawnBit = pawns & (1 << currentBitIndex);

			//OBSTRUCTION CHECK
			if (currentPawnBit == 1) {

				//These special variables are required because the pawn moves are different based on the opponent piece positions
				bool bIsObstructed;
				bool bCanCaptureLeft;
				bool bCanCaptureRight;
				bool bCanPromote;
				bool bCanEnPassantLeft;
				bool bCanEnPassantRight;
				bool bCanMoveTwoSquares;

				unsigned long long pawnsTemp = pawns;
				if (currentBitIndex >= 8 && currentBitIndex <= 15) {
					//DOUBLE SQUARE MOVE
					// Check for Obstruction and Capture (Promotion is not possible in any case if the pawn is able to make a double square move)
					//Isolate the required pawn
					unsigned long long pawnIsolate = 2 << currentBitIndex;
					//Remove it from the copy of the original 
					pawnsTemp = pawnsTemp ^ pawnIsolate;
					//Move the pawn forward 2 squares
					pawnIsolate <<= 16;
					//Add it back to the original
					pawnsTemp = pawnsTemp | pawnIsolate;
					//Append to list
					pawnMoves[pawnChanges] = pawnsTemp;
					pawnChanges++;
					#ifdef DEBUG_MODE
						printf("Pawn at %d can move 2 squares\n", currentBitIndex);
					#endif // Debug prints
				}
				//SINGLE SQUARE MOVE
				/*
				* Check for:
				*	1. Obstruction by another piece
				*	2. Capture
				*	3. Promotion
				*	4. En Passant
				*/
				//Isolate the required pawn
				if (currentBitIndex){
					unsigned long long pawnIsolate = 2 << currentBitIndex;
					//Remove it from the copy of the original 
					pawnsTemp = pawnsTemp ^ pawnIsolate;
					//Move the pawn forward 2 squares
					pawnIsolate <<= 16;
					//Add it back to the original
					pawnsTemp = pawnsTemp | pawnIsolate;
					//Append to list
					pawnMoves[pawnChanges] = pawnsTemp;
					pawnChanges++;
				}
			}



			

			currentBitIndex++;
		}


		

		//ILLEGAL MOVES
		//	If piece's path gets obstructed by another piece
		//	If piece is overlapping with a friendly piece (opponent piece implies capture)
		//	If the side is in check and the move doesn't clear a check

		//DEBUGGING: Print all the moves


		if (amBlackTemp) {
			this->flipSide();
		}
		//TODO: The changes may need to be flipped back as well
		return {
			pawnChanges,
			rookChanges,
			knightChanges,
			bishopChanges,
			queenChanges,
			kingChanges,
			amBlackTemp,

			allChanges
		};
	}
};

