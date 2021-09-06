import numpy as np



def split_dataset(data, ratio=(0.8, 0.1, 0.1), seed=111):
    assert len(ratio) == 3
    length = len(data)
    np.random.seed(seed)
    shuffled_indices = np.random.permutation(length)
    train_num = int(length * ratio[0])
    val_num = int(length * ratio[1])
    train_indices = shuffled_indices[:train_num]
    val_indices = shuffled_indices[train_num : (train_num + val_num)]
    test_indices = shuffled_indices[(train_num + val_num) :]
    print(f"{len(train_indices)} : {len(val_indices)} : {len(test_indices)}")
    return data[train_indices], data[val_indices], data[test_indices]


with open('./labels.txt', 'r') as f:
    all_labels = f.readlines()
labels_npy = np.array(all_labels)
train, _, test = split_dataset(labels_npy, ratio=(0.9, 0, 0.1))
with open('./train.txt', 'w') as f:
    for line in train:
        f.write(line)
with open('./test.txt', 'w') as f:
    for line in test:
        f.write(line)