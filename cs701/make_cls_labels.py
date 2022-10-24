import argparse
import numpy as np

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_label_path", default='/home/hoangle/cs701-wsss/public/train_label.txt', type=str)
    parser.add_argument("--out", default="cls_labels.npy", type=str)
    args = parser.parse_args()

    def to_multi_hot(labels, n_classes=20):
        multi_hot_vector = np.zeros(n_classes)
        for lb in labels:
            multi_hot_vector[lb] = 1
        return multi_hot_vector

    total_label = np.zeros(20)
    d = dict()
    with open(args.train_label_path, 'r') as f:
        for line in f:
            tokens = line.strip().split()
            img_name = tokens[0]
            label = to_multi_hot([int(x) for x in tokens[1:]], n_classes=20)
            d[img_name] = label
            total_label += label

    print(total_label)
    np.save(args.out, d)