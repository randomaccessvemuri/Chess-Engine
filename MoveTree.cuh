#pragma once
#include "State.h"

//TODO
/*
* The move tree node is a node in the move tree, it contains a move and an array of pointers to its children.
*/
class moveTreeNode {
		
};

/*
* To prevent needing to evaluate the same nodes repeatedly, we continously update a move tree and trim it and extend it as the game progresses.
* The move tree is a tree of all possible moves, with the root node being the starting position at first move with the root changing to the current position as the game progresses.
*/
class MoveTree
{
private:
	moveTreeNode* root;

public:
	MoveTree(State boardState);
	~MoveTree();

	//Removes the nodes that are no longer possible from the tree to save memory
	void trimTree();

	// Generates the next layer of the tree (Char because we can realistically only compute about 10-20 layers of the tree at a time so we don't need to use an int)
	void extendTree(char depth);

	//Alpha-Beta Pruning to remove nodes that would not be taken
	void pruneTree();



	


};

