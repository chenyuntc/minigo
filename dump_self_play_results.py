
import torch
import horovod.torch as hvd
hvd.init()
if hvd.rank()>8:
    torch.cuda.set_device(hvd.rank()%2)
else:
    torch.cuda.set_device(0)
from utils.gtp import *
from utils.mcts import *
import glob
import numpy as np
WEIGHTS_PATH='./behaviour_clone_model_weights'

network = Network(BOARD_SIZE)
time_control = TimeControl()
network.trainable(False)
network.load_pt(WEIGHTS_PATH)

files = glob.glob('/d/dlgo/game*.pth')
files = files[hvd.rank()::hvd.size()]

def get_data(board:Board,root_node:Node, explore_th=500):
    policy = np.zeros(81+1) # additional for "PASS"
    N_visits = root_node.visits
    value = root_node.values/root_node.visits
    more_to_explore = []
    for vtx,node in root_node.children.items():
        if node.visits>explore_th:
            more_to_explore.append(node)
        if vtx == -1:
            policy[-1] = node.visits/N_visits    # pass
        else:
            idx = board.vertex_to_index(vtx)
            policy[idx] = node.visits/N_visits           
    features = board.get_features()
    return policy, value,features

import os
from tqdm import tqdm
for file in tqdm(files):
    save_name = file.replace('/game_', '/tree_')
    save_name2 = save_name.replace('/tree_','/data_')
    if os.path.exists((save_name)) or os.path.exists(save_name2):
        continue
    try:
        board = torch.load(file)
        if board.num_passes>=2:
            # print(file)
            continue

        search = Search(board, network, time_control)
        move = search.think(800,0.2,False)
        data = get_data(board, search.root_node)
        torch.save(data, save_name2)
        torch.save(search.root_node, save_name)
    except Exception as e:
        print('eeeeee', e)
        import traceback
        traceback.print_exc()


