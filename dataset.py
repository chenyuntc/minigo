import random

import numpy as np
import torch

import sgf
from board import BLACK, INVLD, NUM_INTESECTIONS, PASS, WHITE, Board
from config import BOARD_SIZE, INPUT_CHANNELS
import dill

def get_symmetry_plane(symm, plane):
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
        self.to_move = None

    def __str__(self):
        out = str()
        out += "policy: {p} | value: {v}\n".format(p=self.policy, v=self.value)
        return out

def do_symmetry(chunk, symm=None):
    assert chunk.policy != None, ""

    if symm == None:
        symm = int(np.random.choice(8, 1)[0])

    for i in range(INPUT_CHANNELS - 2):  # last 2 channels is side to move.
        p = chunk.inputs[i]
        chunk.inputs[i][:][:] = get_symmetry_plane(symm, p)[:][:]

    if chunk.policy != NUM_INTESECTIONS:
        buf = np.zeros(NUM_INTESECTIONS)
        buf[chunk.policy] = 1
        buf = get_symmetry_plane(symm, np.reshape(buf, (BOARD_SIZE, BOARD_SIZE)))
        chunk.policy = int(np.argmax(buf))


class DataSet:
    def __init__(self, dir_name,batch_size, iteration):
        self.buffer = []
        self.batch_size = batch_size
        self.iteration = iteration
        try:
            with open(f"{dir_name}.dill", "rb") as f:
                print('load from preprocssed cache')
                self.buffer = dill.load(f)
        except Exception as e:
            print("it's be slow to parse SGF data for the first time. It will be cached for fast-load after processing")
            self._load_data(dir_name)
            with open(f"{dir_name}.dill", "wb") as f:
                dill.dump(self.buffer, f)

    # Collect training data from sgf dirctor.
    def _load_data(self, dir_name):
        sgf_games = sgf.parse_from_dir(dir_name)
        total = len(sgf_games)
        step = 0
        verbose_step = 1000
        print("total {} games".format(total))
        for game in sgf_games:
            step += 1
            temp = self._process_one_game(game)
            self.buffer.extend(temp)
            if step % verbose_step == 0:
                print("parsed {:.2f}% games".format(100 * step / total))
        if total % verbose_step != 0:
            print("parsed {:.2f}% games".format(100 * step / total))

    # Collect training data from one sgf game.
    def _process_one_game(self, game):
        temp = []
        winner = None
        board = Board(BOARD_SIZE)
        for node in game:
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
                chunk.policy = self._do_text_move(board, color, move)
                temp.append(chunk)

        for chunk in temp:
            if winner == None:
                chunk.value = 0
            elif winner == chunk.to_move:
                chunk.value = 1
            elif winner != chunk.to_move:
                chunk.value = -1
        return temp

    def _do_text_move(self, board, color, move):
        board.to_move = color
        policy = None
        vtx = None
        if len(move) == 0 or move == "tt":
            vtx = PASS
            policy = NUM_INTESECTIONS
        else:
            x = ord(move[0]) - ord("a")
            y = ord(move[1]) - ord("a")
            vtx = board.get_vertex(x, y)
            policy = board.get_index(x, y)
        board.play(vtx)
        return policy

    def __getitem__(self,idx=None):
        return self.get_batch(self.batch_size)

    def __len__(self):
        return self.iteration

    def get_batch(self, batch_size=2048):
        s = random.sample(self.buffer, k=batch_size)
        inputs_batch = []
        policy_batch = []
        value_batch = []
        for chunk in s:
            do_symmetry(chunk)
            inputs_batch.append(chunk.inputs)
            policy_batch.append(chunk.policy)
            value_batch.append([chunk.value])
        inputs_batch = np.array(inputs_batch)
        policy_batch = np.array(policy_batch)
        value_batch = np.array(value_batch)
        return (
            torch.tensor(inputs_batch).float(),
            torch.tensor(policy_batch).long(),
            torch.tensor(value_batch).float(),
        )
