import torch
from sklearn.model_selection import train_test_split

def train_val_test_split(n):
    train, val_test = train_test_split(range(n), test_size=0.5, random_state=42)
    val, test = train_test_split(val_test, test_size=0.5, random_state=42)
    train_mask = torch.full([n], False)
    train_mask[torch.tensor(train)] = True
    val_mask = torch.full([n], False)
    val_mask[torch.tensor(val)] = True
    test_mask = torch.full([n], False)
    test_mask[torch.tensor(test)] = True
    return train_mask, val_mask, test_mask