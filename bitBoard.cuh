/*
	Bitshifting to generate moves:
		Bitshift left 8 times to move the piece down a row w.r.t the printBoard() method
		Bitshift left or right once to move the piece left or right


	The flipSide thing is to not have to write the same code for black and white pieces. It's easier to just flip the board and then flip it back after generating the moves.

*/


#pragma once
#include <stdint.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <thrust/device_vector.h>





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
		bool currentBit = 0; //reused for all move generations

		//PAWN MOVES (At most there can be 8 pawns in a side since you can't promote to a pawn)
		unsigned long long pawnMoves[8];
		while (currentBitIndex!=64) {
			
			currentBit = pawns & (1 << currentBitIndex);

			//This means a pawn was found
			if (currentBit == 1) {
				unsigned long long pawnsTemp = pawns;
				if (currentBitIndex >= 8 && currentBitIndex <= 15) {
					//DOUBLE SQUARE MOVE
					//Isolate the required pawn
					unsigned long long pawnIsolate = pow(2, currentBitIndex);
					//Remove it from the copy of the original 
					pawnsTemp = pawnsTemp ^ pawnIsolate;
					//Move the pawn forward 2 squares
					pawnIsolate <<= 16;
					//Add it back to the original
					pawnsTemp = pawnsTemp | pawnIsolate;

					//DEBUG, REMOVE THIS PLS
					bitBoard temp;
					temp.pawns = pawnsTemp;
					temp.printBoard();
					//Append to list
					pawnMoves[pawnChanges] = pawnsTemp;
					pawnChanges++;
					
				}
				//SINGLE SQUARE MOVE
				//Isolate the required pawn
				unsigned long long pawnIsolate = pow(2, currentBitIndex);
				//Remove it from the copy of the original 
				pawnsTemp = pawnsTemp ^ pawnIsolate;
				//Move the pawn forward 2 squares
				pawnIsolate <<= 16;
				//Add it back to the original
				pawnsTemp = pawnsTemp | pawnIsolate;
				//Append to list
				pawnMoves[pawnChanges] = pawnsTemp;
				pawnChanges++;

				//DEBUG, REMOVE THIS PLS
				bitBoard temp;
				temp.pawns = pawnsTemp;
				temp.printBoard();
			}

			

			currentBitIndex++;
		}


		//ROOK MOVES

		 //KNIGHT MOVES
		currentBit = 0;


		//We need to generate all possible knight moves. We do this by generating all possible knight moves for each knight and then ORing them together.
		

		//BISHOP MOVES

		//QUEEN MOVES

		//KING MOVES


		/*
		* CHECKING LEGALITY OF EACH MOVE!
		*/



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

