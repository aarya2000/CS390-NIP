import os
import matplotlib.pyplot as plt

vals = ['mnist_d', 'mnist_f', 'cifar_10', 'cifar_100_c', 'cifar_100_f']
ann_list = []

for val in vals:
    temp = val + '_ann'
    ann_list.append(temp)

parent_dir = os.getcwd()

cnn_acc = []

for val in vals:
    temp = os.path.join(parent_dir, val)
    final = os.path.join(temp, 'acc')
    acc_f = open(final, 'r')
    old_acc = acc_f.read()
    acc = float(old_acc)
    cnn_acc.append(acc)
    acc_f.close()

plt.bar(vals, cnn_acc)
xlocs, xlabs = plt.xticks()
xlocs = [i+1 for i in range(0, 10)]
xlabs = [i/2 for i in range(0, 10)]
plt.title('CNN model')
plt.xlabel('Dataset')
plt.ylabel('Accuracy')
for i, v in enumerate(cnn_acc):
    plt.text(xlocs[i] - 1.25, v + 0.01, str(v))
plt.show()

ann_acc = []

for val in ann_list:
    temp = os.path.join(parent_dir, val)
    final = os.path.join(temp, 'acc')
    acc_f = open(final, 'r')
    old_acc = acc_f.read()
    acc = float(old_acc)
    ann_acc.append(acc)
    acc_f.close()

plt.bar(vals, ann_acc)
xlocs, xlabs = plt.xticks()
xlocs = [i+1 for i in range(0, 10)]
xlabs = [i/2 for i in range(0, 10)]
plt.title('ANN model')
plt.xlabel('Dataset')
plt.ylabel('Accuracy')
for i, v in enumerate(ann_acc):
    plt.text(xlocs[i] - 1.25, v + 0.01, str(v))
plt.show()

