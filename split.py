import os
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

import numpy as np


class_path = 'public/class.txt'
label_path = 'public/train_label.txt'
split_dir = 'public/split'

def read_label(path):
    image_labels = {}
    with open(path, 'r') as f:
        for line in f:
            tokens = line.strip().split()
            image_labels[tokens[0]] = [int(label) for label in tokens[1:]]
    return image_labels

def read_class(path):
    classes = {}
    with open(path, 'r') as f:
        for line in f:
            tokens = line.strip().split()
            classes[int(tokens[0])] = tokens[1]
    return classes

train_labels = read_label(label_path)
classes = read_class(class_path)
labels = list(classes.values())

print('#images =', len(train_labels))
df = pd.DataFrame.from_dict(data=list(train_labels.keys()), orient='columns')
df = df.rename(columns={0: 'image'})
df['labels'] = df['image'].apply(lambda x: [classes[t_label] for t_label in train_labels[x]])
text_to_category = {label:[] for label in classes.values()}
for idx, item in df.iterrows():
    for label in text_to_category:
        if label in item['labels']:
            text_to_category[label].append(1)
        else:
            text_to_category[label].append(0)

for label in text_to_category:
    df[label] = text_to_category[label]

X, Y = df['image'].to_numpy(), df[labels].to_numpy(dtype=np.float32)

msss = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=1234)

for inc, (train_index, test_index) in enumerate(msss.split(X, Y)):
    print("TRAIN:", len(train_index), train_index, "TEST:", len(test_index), test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]
    os.makedirs(os.path.join(split_dir, '{}'.format(inc)), exist_ok=True)
    with open(os.path.join(split_dir, '{}/train.txt'.format(inc)), 'w') as f:
        for img_name in X_train:
            f.write('{} {}\n'.format(img_name, ' '.join([str(s) for s in train_labels[img_name]])))
    with open(os.path.join(split_dir, '{}/validation.txt'.format(inc)), 'w') as f:
        for img_name in X_test:
            f.write('{} {}\n'.format(img_name, ' '.join([str(s) for s in train_labels[img_name]])))

