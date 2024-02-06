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

	__device__ __host__ bool checkOverlap(unsigned long long pieceRepresentation) {
		return (pawns & pieceRepresentation) | (rooks & pieceRepresentation) | (knights & pieceRepresentation) | (bishops & pieceRepresentation) | (queens & pieceRepresentation) | (kings & pieceRepresentation);
	}

	//TODO
	//Takes an index and checks if a piece can move to that index. Used to prevent king from moving into a position that is being attacked by the opponent
	__device__ __host__ bool checkIfIndexIsSeen(int index) {
	//Rook

}



	/// <summary>
	/// Generates all possible moves for the current board state. Returns a vector of bitBoards that represent all possible moves. This can be at most 218 moves.
	/// </summary>
	/// <TODO>
	/// - Eliminate branching as much as possible
	/// - Add King Check
	/// </TODO>
	/// <param name="opponentBoard">The opponent's board state: Useful for pawn diagonal capture, eliminating obstructed bishop, rook and queen moves etc. </param>
	__device__ internals::cudaVector<bitBoard> generateAllStates(bitBoard opponentBoard) {

		internals::cudaVector<bitBoard> allMoves(218);
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

		int currentBitIndex = 0; //reused for all move generations
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
			currentPawnBit = pawns & ((unsigned long long) 1 << currentBitIndex);
			if (currentPawnBit == 1) {

				//These special variables are required because the pawn moves are different based on the opponent piece positions
				
				//ANDing all of them to combine them into one variable
				unsigned long long opponentAllCombined = opponentBoard.pawns & opponentBoard.rooks & opponentBoard.knights & opponentBoard.bishops & opponentBoard.queens;

				//Check if the pawn is obstructed by another piece
				bool bIsObstructed = currentBitIndex + 8 & opponentAllCombined;

				//Check if pawn is checking the king
				bool bIsCheckingLeft = currentBitIndex + 7 & opponentBoard.kings;
				bool bIsCheckingRight = currentBitIndex + 9 & opponentBoard.kings;

				//Check if pawn is capturing a piece
				bool bCanCaptureLeft = currentBitIndex + 7 & opponentAllCombined;
				bool bCanCaptureRight = currentBitIndex + 9 & opponentAllCombined;

				bool bCanPromote = currentBitIndex + 8 > 63;

				//This is obiviously wrong but I'll tackle it later (TODO)
				bool bCanEnPassantLeft = false;
				bool bCanEnPassantRight = false;
				// Check if pawn is in the 2nd row and the 3rd row is empty
				bool bCanMoveTwoSquares = currentBitIndex < 16 && currentBitIndex + 16 & opponentAllCombined;


				//We can't have overlapping pieces so no point evaluating all the pieces for the same location
				continue;
			}



			//ROOK MOVES
			currentRookBit = rooks & ((unsigned long long) 1 << currentBitIndex);
			if (currentRookBit == 1) {
				unsigned long long defaultRookPlace = ((unsigned long long)1 << currentBitIndex);
				unsigned long long currentRookPlace = defaultRookPlace;
				
				//Add all moves towards the left
				//Check for opponent and rook intersection as well as rook and friendly intersection
				while (!(opponentBoard.checkOverlap(currentRookPlace) && this->checkOverlap(currentRookPlace))) {
					//This is placed above so that the last move overlaps with the opponent's piece, implying a capture
					rookMoves.push_back(currentRookPlace);
					rookChanges++;
					currentRookPlace = currentRookPlace << 1;
				}

				//Add all moves towards the right
				currentRookPlace = defaultRookPlace;
				while (!(opponentBoard.checkOverlap(currentRookPlace) && this->checkOverlap(currentRookPlace))) {
					rookMoves.push_back(currentRookPlace);
					rookChanges++;
					currentRookPlace = currentRookPlace >> 1;
				}

				//Add all moves towards the top
				currentRookPlace = defaultRookPlace;
				while(!(opponentBoard.checkOverlap(currentRookPlace) && this->checkOverlap(currentRookPlace))) {
					rookMoves.push_back(currentRookPlace);
					rookChanges++;
					currentRookPlace = currentRookPlace << 8;
				}

				//Add all moves towards the bottom
				currentRookPlace = defaultRookPlace;
				while (!(opponentBoard.checkOverlap(currentRookPlace) && this->checkOverlap(currentRookPlace))) {
					rookMoves.push_back(currentRookPlace);
					rookChanges++;
					currentRookPlace = currentRookPlace >> 8;
				}
			}
			
			//KNIGHT MOVES
			currentKnightBit = knights & ((unsigned long long) 1 << currentBitIndex);

			if (currentKnightBit) {
				unsigned long long defaultKnightPlace = ((unsigned long long)1 << currentBitIndex);
				unsigned long long currentKnightPlace = defaultKnightPlace;

				internals::cudaVector<unsigned long long> knightMoves(8);

				knightMoves.push_back(currentKnightPlace << 17); //Top right
				knightMoves.push_back(currentKnightPlace << 15); // Top left
				knightMoves.push_back(currentKnightPlace << 10); // Right top
				knightMoves.push_back(currentKnightPlace << 6); // Left top

				knightMoves.push_back(currentKnightPlace >> 17); //Bottom left
				knightMoves.push_back(currentKnightPlace >> 15); // Bottom right
				knightMoves.push_back(currentKnightPlace >> 10); // Left bottom
				knightMoves.push_back(currentKnightPlace >> 6); // Right bottom

				//Remove all moves that are out of bounds or obstructed by friendly/opponent pieces
				for (int i = 0; i < knightMoves.getSize(); i++) {
					if (knightMoves[i] >= 0 && knightMoves[i] < 64) {
						if (!(opponentBoard.checkOverlap(knightMoves[i]) && this->checkOverlap(knightMoves[i]))) {
							knightMoves.push_back(knightMoves[i]);
							knightChanges++;
						}
					}
				}				
				continue;
			}

			//BISHOP MOVES
			currentBishopBit = bishops & ((unsigned long long) 1 << currentBitIndex);	

			if (currentBishopBit) {
				unsigned long long defaultBishopPlace = ((unsigned long long)1 << currentBitIndex);
				unsigned long long currentBishopPlace = defaultBishopPlace;

				internals::cudaVector<unsigned long long> bishopMoves(13);

				//Add all moves towards the top right
				while (!(opponentBoard.checkOverlap(currentBishopPlace) && this->checkOverlap(currentBishopPlace))) {
					bishopMoves.push_back(currentBishopPlace);
					bishopChanges++;
					currentBishopPlace = currentBishopPlace << 7;
				}

				//Add all moves towards the top left
				currentBishopPlace = defaultBishopPlace;
				while (!(opponentBoard.checkOverlap(currentBishopPlace) && this->checkOverlap(currentBishopPlace))) {
					bishopMoves.push_back(currentBishopPlace);
					bishopChanges++;
					currentBishopPlace = currentBishopPlace << 9;
				}

				//Add all moves towards the bottom right
				currentBishopPlace = defaultBishopPlace;
				while (!(opponentBoard.checkOverlap(currentBishopPlace) && this->checkOverlap(currentBishopPlace))) {
					bishopMoves.push_back(currentBishopPlace);
					bishopChanges++;
					currentBishopPlace = currentBishopPlace >> 9;
				}

				//Add all moves towards the bottom left
				currentBishopPlace = defaultBishopPlace;
				while (!(opponentBoard.checkOverlap(currentBishopPlace) && this->checkOverlap(currentBishopPlace))) {
					bishopMoves.push_back(currentBishopPlace);
					bishopChanges++;
					currentBishopPlace = currentBishopPlace >> 7;
				}

				continue;
			}

			//QUEEN MOVES
			currentQueenBit = queens & ((unsigned long long) 1 << currentBitIndex);
			if (currentQueenBit) {
				unsigned long long defaultQueenPlace = ((unsigned long long)1 << currentBitIndex);
				unsigned long long currentQueenPlace = defaultQueenPlace;

				//Add all moves towards the left
				//Check for opponent and rook intersection as well as rook and friendly intersection
				while (!(opponentBoard.checkOverlap(currentQueenPlace) && this->checkOverlap(currentQueenPlace))) {
					rookMoves.push_back(currentQueenPlace);
					rookChanges++;
					currentQueenPlace = currentQueenPlace << 1;
				}

				//Add all moves towards the right
				currentQueenPlace = defaultQueenPlace;
				while (!(opponentBoard.checkOverlap(currentQueenPlace) && this->checkOverlap(currentQueenPlace))) {
					rookMoves.push_back(currentQueenPlace);
					rookChanges++;
					currentQueenPlace = currentQueenPlace >> 1;
				}

				//Add all moves towards the top
				currentQueenPlace = defaultQueenPlace;
				while (!(opponentBoard.checkOverlap(currentQueenPlace) && this->checkOverlap(currentQueenPlace))) {
					rookMoves.push_back(currentQueenPlace);
					rookChanges++;
					currentQueenPlace = currentQueenPlace << 8;
				}

				//Add all moves towards the bottom
				currentQueenPlace = defaultQueenPlace;
				while (!(opponentBoard.checkOverlap(currentQueenPlace) && this->checkOverlap(currentQueenPlace))) {
					rookMoves.push_back(currentQueenPlace);
					rookChanges++;
					currentQueenPlace = currentQueenPlace >> 8;
				}

				//Add all moves towards the top right
				while (!(opponentBoard.checkOverlap(currentQueenPlace) && this->checkOverlap(currentQueenPlace))) {
					bishopMoves.push_back(currentQueenPlace);
					bishopChanges++;
					currentQueenPlace = currentQueenPlace << 7;
				}

				//Add all moves towards the top left
				currentQueenPlace = defaultQueenPlace;
				while (!(opponentBoard.checkOverlap(currentQueenPlace) && this->checkOverlap(currentQueenPlace))) {
					bishopMoves.push_back(currentQueenPlace);
					bishopChanges++;
					currentQueenPlace = currentQueenPlace << 9;
				}

				//Add all moves towards the bottom right
				currentQueenPlace = defaultQueenPlace;
				while (!(opponentBoard.checkOverlap(currentQueenPlace) && this->checkOverlap(currentQueenPlace))) {
					bishopMoves.push_back(currentQueenPlace);
					bishopChanges++;
					currentQueenPlace = currentQueenPlace >> 9;
				}

				//Add all moves towards the bottom left
				currentQueenPlace = defaultQueenPlace;
				while (!(opponentBoard.checkOverlap(currentQueenPlace) && this->checkOverlap(currentQueenPlace))) {
					bishopMoves.push_back(currentQueenPlace);
					bishopChanges++;
					currentQueenPlace = currentQueenPlace >> 7;
				}
			}

			//KING MOVES
			//This is a bit more complicated because the king can't move into a position that is being attacked by the opponent
			currentKingBit = kings & ((unsigned long long) 1 << currentBitIndex);
			if (currentKingBit) {
				unsigned long long defaultKingPlace = ((unsigned long long)1 << currentBitIndex);
				unsigned long long currentKingPlace = defaultKingPlace;

				internals::cudaVector<unsigned long long> kingMoves(8);

				kingMoves.push_back(currentKingPlace << 1); //Right
				kingMoves.push_back(currentKingPlace >> 1); //Left
				kingMoves.push_back(currentKingPlace << 8); //Top
				kingMoves.push_back(currentKingPlace >> 8); //Bottom

				kingMoves.push_back(currentKingPlace << 7); //Top right
				kingMoves.push_back(currentKingPlace << 9); //Top left
				kingMoves.push_back(currentKingPlace >> 7); //Bottom right
				kingMoves.push_back(currentKingPlace >> 9); //Bottom left

				//Remove all moves that are out of bounds or obstructed by friendly/opponent pieces
				for (int i = 0; i < kingMoves.getSize(); i++) {
					//Bounds check
					if (kingMoves[i] >= 0 && kingMoves[i] < 64) {
						//Overlap check (SPECIAL: Also check if the move is being attacked by the opponent)
						if (!(opponentBoard.checkOverlap(kingMoves[i]) && this->checkOverlap(kingMoves[i])) && !opponentBoard.checkIfIndexIsSeen(kingMoves[i])) {
							kingMoves.push_back(kingMoves[i]);
							kingChanges++;
						}
					}
				}
				continue;
			}
			currentBitIndex++;
		}


		//TODO: Process captures and illegal moves

		//ILLEGAL MOVES
		//	If piece's path gets obstructed by another piece
		//	If piece is overlapping with a friendly piece (opponent piece implies capture)
		//	If the side is in check and the move doesn't clear a check

		//DEBUGGING: Print all the moves




		if (amBlackTemp) {
			this->flipSide();
		}
		//TODO: The changes may need to be flipped back as well
		return allMoves;
	}
};

