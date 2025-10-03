import os

import numpy as np
from torch.utils.data import Dataset

class Shakespeare(Dataset):
    def __init__(self, data_dir, target_size, train, context_len):
        """
        target_size used to fake the gradient accumulation steps
        
        target_size = batch_size * num_accumulation_steps
        """
        name = "train" if train else "val"
        self.data = np.memmap(os.path.join(data_dir, f'{name}.bin'), dtype=np.uint16, mode='r')
        
        self.target_size = target_size
        self.context_len = context_len

    def __len__(self):
        return self.target_size

    def __getitem__(self, idx):
        start_idx = np.random.randint(0, len(self.data) - self.context_len - 1)
        x = self.data[start_idx:start_idx + self.context_len]
        return x
        
