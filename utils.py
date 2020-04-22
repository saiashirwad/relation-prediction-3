import numpy as np 
import torch 

def negative_sampling(pos_samples, num_entity, negative_rate):
    if pos_samples.shape[0] == 3:
        pos_samples = pos_samples.t()
    size_of_batch = len(pos_samples)
    num_to_generate = size_of_batch * negative_rate
    neg_samples = np.tile(pos_samples, (negative_rate, 1))
    # print(neg_samples.shape, "negative_samples")
    labels = np.zeros(size_of_batch * (negative_rate + 1), dtype=np.float32)
    labels[: size_of_batch] = 1
    values = np.random.randint(num_entity, size=num_to_generate)
    choices = np.random.uniform(size=num_to_generate)
    subj = choices > 0.5
    obj = choices <= 0.5
    neg_samples[subj, 0] = values[subj]
    neg_samples[obj, 1] = values[obj]

    return torch.from_numpy(np.concatenate((pos_samples, neg_samples))), torch.from_numpy(labels)