import numpy as np
import collections

num_feature = int(input('Enter Number of Features - '))

X = []
for i in range(num_feature):
    a = list(input(f'Enter values for Feature X{i} (space-separated) : ').split())
    X.append(a)

Y = list(input('Enter values for Feature Y (space-separated) : ').split())
n = len(Y)

for v in range(num_feature):
    if n != len(X[v]):
        print('Error : number of values should be equal in each feature')

X_rows = list(zip(*X))
data = list(zip(X_rows, Y))

def entropy(counter, total):
    ent = 0
    for count in counter.values():
        p = count / total
        if p != 0:
            ent += -p * np.log2(p)
    return ent

f_idx = 0
value_count = collections.defaultdict(list)

for f_idx in range(num_feature):
    value_count = collections.defaultdict(list)
    
    for f, t in data:
        F = f[f_idx]
        value_count[F].append(t)

    conditional_entropy = 0
    for feature, target in value_count.items():
        counter = collections.Counter(target)
        size = sum(counter.values())
        x_entropy = entropy(counter, size)
        conditional_entropy += (size / n) * x_entropy

    cnt_Y = collections.Counter(Y)
    Y_entropy = entropy(cnt_Y, n)
    ig = Y_entropy - conditional_entropy
    print(f"Information Gain for Feature X{f_idx}: {ig}")