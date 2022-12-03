import json
from collections import Counter
import numpy as np

class Sequence_count_encoder:
    '''
    Кодирует входящую последовательность в последовательность из чисел
    (самые частотнеые элементы последовательности кодируются меньшими числами)
    '''
    char_counts = []
    sorted_chars = []
    char_to_idx = {}
    idx_to_char = {}
    
    def create_from_txt(self, file_path):
        with open(file_path, encoding="utf-8") as f:
            lines = f.readlines()
        text_sample = ' '.join(lines)
        self.create_from_string(text_sample)

    def load_from_json(self, file_path, num_lines = 100):
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)

        lines = []
        for d in data[:num_lines]:
            lines.append(d["content"])
        text_sample = ' '.join(lines)
        self.create_from_string(text_sample)

    def create_from_string(self, text_sample):
        self.char_counts = Counter(text_sample)
        self.char_counts = sorted(self.char_counts.items(), key = lambda x: x[1], reverse=True)

        self.sorted_chars = [char for char, _ in self.char_counts]
        self.char_to_idx = {char: index for index, char in enumerate(self.sorted_chars)}
        self.idx_to_char = {v: k for k, v in self.char_to_idx.items()}
        
    def text_to_seq(self, text_sample):
        sequence = np.array([self.char_to_idx.get(char, 0) for char in text_sample])
        return sequence