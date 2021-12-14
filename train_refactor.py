import time

import torch
import torch.nn as nn
import torch.optim as optim

from network import Network
from dataset import DataSet


def main(
    data_dir="sgf", steps=400000, verbose_step=1000, batch_size=2048, learning_rate=1e-3
):
    network = Network(board_size=9)
    network.trainable()

    # Prepare the data set from sgf files.
    data_set = DataSet(data_dir, batch_size, steps)
    dataloader = torch.utils.data.DataLoader(
        data_set,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda x: x[0],
        num_workers=8,
    )

    cross_entry = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()
    optimizer = optim.Adam(network.parameters(), lr=learning_rate, weight_decay=1e-4)
    p_running_loss = 0
    v_running_loss = 0
    running_loss_record = []
    clock_time = time.time()
    from tqdm import tqdm

    for step, data in tqdm(enumerate(dataloader)):
        #  get the batch data.
        inputs, target_p, target_v = data  # data_set.get_batch(batch_size)
        inputs = inputs.cuda()
        target_p = target_p.cuda()
        target_v = target_v.cuda()

        # Forward with network.
        p, v = network(inputs)

        # compute loss result and update network.
        p_loss = cross_entry(p, target_p)
        v_loss = mse_loss(v, target_v)
        loss = p_loss + v_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # metrics
        p_running_loss += p_loss.item()
        v_running_loss += v_loss.item()

        # learning rate decay
        if (step + 1) % 128000 == 0:
            for g in optimizer.param_groups:
                g["lr"] *= 0.1
            network.save_pt(str(step))

        #  print verbose.
        if (step + 1) % verbose_step == 0:
            elapsed = time.time() - clock_time
            rate = verbose_step / elapsed
            remaining_step = steps - step
            estimate_remaining_time = int(remaining_step / rate)
            print(
                f"{time.strftime('%H:%M:%S')} steps: {step + 1}/{steps}, {100 * ((step + 1) / steps):.2f}% -> policy loss: {p_running_loss / verbose_step:.4f}, value loss: {v_running_loss / verbose_step:.4f} | rate: {rate:.2f}(step/sec), estimate: {estimate_remaining_time}(sec)"
            )
            running_loss_record.append(
                (
                    step + 1,
                    p_running_loss / verbose_step,
                    v_running_loss / verbose_step,
                )
            )
            p_running_loss = 0
            v_running_loss = 0
            clock_time = time.time()

    print("Trainig is over.")


if __name__ == "__main__":
    import fire
    fire.Fire(main)