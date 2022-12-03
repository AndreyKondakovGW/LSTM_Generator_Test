import numpy as np
import torch


class MyLoader:
    def __init__(self, train_seq, batch_size=16, seq_length=64):
        self.train_seq = train_seq
        self.batch_size = batch_size
        self.seq_length = seq_length

    def get_batch(self):
        '''
        Возвращает батч из последовательности train/test
        На выходе train тензор размера (batch_size, seq_len, 1)
        '''
        trains = []
        targets = []
        for _ in range(self.batch_size):
            # выбираем случайную последовательность длины seq_length
            batch_start = np.random.randint(0, len(self.train_seq) - self.seq_length)
            chunk = self.train_seq[batch_start: batch_start + self.seq_length]

            # рассматриваем последний элемент как целевой
            train = torch.LongTensor(chunk[:-1]).view(-1, 1)
            target = torch.LongTensor(chunk[1:]).view(-1, 1)
            trains.append(train)
            targets.append(target)
        return torch.stack(trains, dim=0), torch.stack(targets, dim=0)