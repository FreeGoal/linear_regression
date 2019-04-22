import numpy

f = open("data.csv", 'r')

x = []
y = []
for line in f.readlines():
    row = line.split(',')
    x.append(row[1 : -1])
    y.append(row[-1])

train_size = int(0.8 * len(x))
test_size = int(0.2 * len(x))

train_x = x[0 : train_size]
train_y = y[0 : train_size]
test_x = x[-test_size]
test_y = y[-test_size]

print("tr: %d, ts: %d" % (train_size, test_size))