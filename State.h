#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "bitBoard.cuh"
/// <summary>
/// Holds the state of the game. This includes the bitboards for each player, and whose turn it is, whether one is in check, etc.
/// </summary>
class State
{
	bitBoard black;
	bitBoard white;
	bool turn; // true = white, false = black
public:
	/// <summary>
	/// Initializes two bitboards, one for each player. Flips the second one to represent the other player.
	/// </summary>
	State() {
		white = bitBoard();
		black = bitBoard();
		black.flipSide();
		turn = true; //white goes first
	}

	/// <summary>
	/// Takes a move and makes it on the board. This is only for user input, not for the AI.
	/// </summary>
	/// <param name="move">
	/// <para>: A three character string in<para>
	/// </param>
	void makeMove(char* move) {
		
			
	}


	
};

