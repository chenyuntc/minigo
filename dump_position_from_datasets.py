from dataset import *
import pandas as pd
import os
save_dir = '/d/dlgo/'
os.makedirs(save_dir,exist_ok=True)
import horovod.torch as hvd
hvd.init()

dir_name='./sgf'
sgf_games = sgf.parse_from_dir(dir_name)
for idx, sgf_game in enumerate(sgf_games):
    sgf_game.idx=idx
total = len(sgf_games)
def text_move(board:Board, color, move):
    board.to_move = color
    policy = None
    vtx = None
    if len(move) == 0 or move == "tt":
        vtx = PASS
        policy = NUM_INTESECTIONS
    else:
        x = ord(move[0]) - ord('a')
        y = ord(move[1]) - ord('a')
        vtx = board.get_vertex(x, y)
        policy = board.get_index(x, y)
    board.play(vtx)
    return policy
def process_one_sgf(game):
    board = Board(BOARD_SIZE)
    for idx,node in enumerate(game):
        color = INVLD
        move = None
        if "W" in node.properties:
            color = WHITE
            move = node.properties["W"][0]
        elif "B" in node.properties:
            color = BLACK
            move = node.properties["B"][0]
        elif "RE" in node.properties:
            result = node.properties["RE"][0]
            if "B+" in result:
                winner = BLACK
            elif "W+" in result:
                winner = WHITE
        if color != INVLD:
            chunk = Chunk()
            chunk.inputs = board.get_features()
            chunk.to_move = color
            chunk.policy = text_move(board, color, move)
            if np.random.rand()>0.1:
                continue
            try:
                torch.save((board), f"{save_dir}/game_{game.idx:06d}_step_{idx:02d}.pth")
                # pd.to_pickle(board, f"{save_dir}/game_{game.idx:06d}_step_{idx:02d}pkl")
            except Exception as e:
                print(e, game.idx)

import multiprocessing as mp
pool = mp.Pool(24)
from tqdm import tqdm
sgf_games =  sgf_games[hvd.rank()::hvd.size()]
temps = list(tqdm(map(process_one_sgf,sgf_games), total=len(sgf_games)))