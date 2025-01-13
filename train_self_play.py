import os

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"


import glob
import os
import time
from functools import lru_cache

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from utils.board import *
from utils.mcts import *


class Dataset:
    def __init__(self, batch_size=64):
        self.files = glob.glob("/d/dlgo/tree*")
        print("num data", len(self.files))
        self.batch_size = batch_size

    @lru_cache(maxsize=100000)
    def get_datax(self, filename):
        data_path = filename.replace("tree", "data")
        if os.path.exists(data_path):
            policy, value, features = torch.load(data_path)
        else:
            board = filename.replace("/tree", "/game")
            board = torch.load(board)
            tree = torch.load(filename)
            policy, value, features = self.get_data(board, tree)
        chunk = Chunk()
        chunk.inputs = features
        chunk.policy = policy
        chunk.value = value
        chunk.value = 0  # doesn't matter
        return chunk

    def __getitem__(self, idx):
        if idx % 1000 == 0:
            self.files = glob.glob("/d/dlgo/tree*")
            print("num data", len(self.files))
        return self.get_batch()

    def get_batch(self):
        idxs = np.random.choice(len(self.files), self.batch_size)
        datas = []
        for idx in idxs:
            try:
                filename = self.files[idx]
                datas.append(self.get_datax(filename))
            except Exception as e:
                pass
        inputs_batch = []
        policy_batch = []
        value_batch = []
        for chunk_ in datas:
            import copy
            for sym_idx in range(8):
                chunk = copy.deepcopy(chunk_)
                chunk.do_symmetry(sym_idx)
                inputs_batch.append(chunk.inputs)
                policy_batch.append(chunk.policy)
                value_batch.append([chunk.value])
        return (
            torch.tensor(inputs_batch).float(),
            torch.tensor(policy_batch).float(),
            torch.tensor(value_batch).float(),
        )

    def get_data(self, board: Board, root_node: Node, explore_th=500):
        policy = np.zeros(81 + 1)  # additional for "PASS"
        N_visits = root_node.visits
        value = root_node.values / root_node.visits
        more_to_explore = []
        for vtx, node in root_node.children.items():
            if node.visits > explore_th:
                more_to_explore.append(node)
            if vtx == -1:
                policy[-1] = node.visits / N_visits  # pass
            else:
                idx = board.vertex_to_index(vtx)
                policy[idx] = node.visits / N_visits
        features = board.get_features()
        return policy, value, features

    def __len__(self):
        return len(self.files) * 4


def symmetry_board(symm, plane):
    use_flip = False
    if symm // 4 != 0:
        use_flip = True
    symm = symm % 4
    transformed = np.rot90(plane, symm)
    if use_flip:
        transformed = np.flip(transformed, 1)
    return transformed


class Chunk:
    def __init__(self):
        self.inputs = None
        self.policy = None
        self.value = None

    def __str__(self):
        out = str()
        out += "policy: {p} | value: {v}\n".format(p=self.policy, v=self.value)
        return out

    def do_symmetry(self, symm=None):
        assert self.policy is not None, ""

        if symm == None:
            symm = int(np.random.choice(8, 1)[0])
        for i in range(INPUT_CHANNELS - 2):  # last 2 channels is side to move.
            p = self.inputs[i]
            self.inputs[i][:][:] = symmetry_board(symm, p)[:][:]
        buf = self.policy[:NUM_INTESECTIONS].copy()
        buf = symmetry_board(symm, np.reshape(buf, (BOARD_SIZE, BOARD_SIZE)))
        self.policy[:NUM_INTESECTIONS] = buf.reshape(-1)


def cross_entropy(outputs, targets):
    """Calculate crossentropy between two distribution """
    # import ipdb; ipdb.set_trace()
    outputs = torch.nn.functional.log_softmax(outputs, dim=1)  # normalize input
    acc = (targets.max(1)[1] == outputs.max(1)[1]).sum().item() / (targets.shape[0])
    return -torch.sum(targets * outputs) / targets.size()[0], acc


class TrainingPipe:
    def __init__(self):
        self.network = Network(BOARD_SIZE)
        self.network.trainable()

        # Prepare the data set from sgf files.
        self.data_set = Dataset()
        self.dataloader = torch.utils.data.DataLoader(
            self.data_set,
            batch_size=1,
            shuffle=False,
            collate_fn=lambda x: x[0],
            num_workers=16,
        )

    def running(self, steps=1000000, verbose_step=1000, learning_rate=1e-5):
        optimizer = optim.Adam(
            self.network.parameters(), lr=learning_rate, weight_decay=1e-4
        )
        policy_running_loss = 0
        accuracy = 0
        start_time = time.time()

        for step, data in tqdm(enumerate(self.dataloader)):
            for _ in range(2):
                inputs, target_p, _ = data  # self.data_set.get_batch(batch_size)
                inputs = inputs.cuda()
                target_p = target_p.cuda()

                policy_output, _ = self.network(inputs)
                p_loss, acc = cross_entropy(policy_output, target_p)
                loss = p_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                policy_running_loss += p_loss.item()
                accuracy += acc

            if (step + 1) % verbose_step == 0:
                elapsed = time.time() - start_time
                rate = verbose_step / elapsed
                remaining_step = steps - step
                estimate_remaining_time = int(remaining_step / rate)
                print(
                    f"{time.strftime('%H:%M:%S')} steps: {step + 1}/{steps}, {100 * ((step + 1) / steps):.2f}% -> policy loss: {policy_running_loss / verbose_step:.4f}, accuracy: {accuracy / verbose_step:.4f} | rate: {rate:.2f}(step/sec), estimate: {estimate_remaining_time}(sec)"
                )
                policy_running_loss = 0
                accuracy = 0
                start_time = time.time()
            if (step + 1) % 1000 == 0:
                self.save_weights("selfplay_w_" + str(step))
            if (step + 1) % 2500 == 0:
                for g in optimizer.param_groups:
                    g["lr"] *= 0.1

    def save_weights(self, name):
        self.network.save_ckpt(name)

    def load_weights(self, name):
        if name != None:
            self.network.load_ckpt(name)

pipe = TrainingPipe()
pipe.load_weights("./behaviour_clone_model_weights") # best behaviour cloning model
pipe.running(12)
