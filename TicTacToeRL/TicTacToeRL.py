from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import tensorflow as tf
import numpy as np

#RL rewards
REWARD_WIN = 1.
REWARD_DRAW = 0.
REWARD_ACTION = 0.
REWARD_LOOSE = -1.
REWARD_INVALID = -1.

action_size=9

def emptyboard():
    board = np.array([[0,0,0],[0,0,0],[0,0,0]])
    return board

def checkSet(a,s):
    x = a % 3
    print(x.shape)
    y = a // 3
    return not s[x,y] == 0

def setMachine(a, s, random):
    x = a % 3
    y = a // 3
    board, success = setToken(x,y, s)
    if not success :
        return board, REWARD_INVALID, True
    else:
        winner = checkWin(board)
        if winner == 1:
            return board, REWARD_WIN, True
        if winner == 0:
            if checkDraw(board):
                return board, REWARD_DRAW, True
        if winner == -1:
            return board, REWARD_LOOSE, True
        if random:
            board = setRandom(board)
            winner = checkWin(board)
            if winner == -1:
                return board, REWARD_LOOSE, True
        return board, REWARD_ACTION, False


def setRandom(s):
    val = 1
    while not val == 0:
        x = np.random.randint(0,3)
        y = np.random.randint(0,3)
        val = s[x,y]
    board, success = setToken(x,y,s)
    return board

def setToken(x, y, board):
    if 2 < x  or x < 0:
        return board, False
    if 2 < y or y < 0:
        return board, False

    countx = np.count_nonzero(board == 1)
    county = np.count_nonzero(board == -1)

    value = 1
    if countx > county:
        value = -1
    
    if not board[x,y] == 0:
        board[x,y] = value
        return board, False

    board[x,y] = value
    return board, True

def checkDraw(board):
    return np.count_nonzero(board) == 9
 

def checkWin(board):
    for row in board:
        if len(set(row)) == 1:
            return row[0]
        if len(set([board[i][i] for i in range(len(board))])) == 1:
            return board[0][0]
        if len(set([board[i][len(board)-i-1] for i in range(len(board))])) == 1:
            return board[0][len(board)-1]
    return 0

def printboard(board):
    print(board)
