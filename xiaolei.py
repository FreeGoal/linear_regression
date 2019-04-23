import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from matplotlib.font_manager import FontProperties

font = FontProperties(fname=r"c:\windows\fonts\msyh.ttc", size=10)
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签

def runplt(size=None):
    plt.figure(figsize=size)
    plt.title('匹萨价格与直径数据',fontproperties=font)
    plt.xlabel('直径（英寸）',fontproperties=font)
    plt.ylabel('价格（美元）',fontproperties=font)
    plt.axis([300, 350, 300, 350])
    plt.grid(True)
    return plt

f = open("data.csv", 'r')

x = []
y = []
for line in f.readlines():
    row = line.split(',')
    for i in range(1, len(row) - 1):
        row[i] = float(row[i])
    x.append(row[1:-2])
    y.append(float(row[-1][0:-2]))

train_size = int(0.8 * len(x))
test_size = int(0.2 * len(x))

train_x = np.array(x[0 : train_size])
train_y = np.array(y[0 : train_size])
test_x = np.array(x[-test_size:-1])
test_y = np.array(y[-test_size:-1])

print("tr: %d, ts: %d" % (train_size, test_size))

X_train = train_x[:,0].reshape(-1,1).tolist()
y_train = train_y.reshape(-1,1).tolist()
print(y_train[0:10])
X_test = test_x[:,0].reshape(-1,1).tolist()
y_test = test_y.reshape(-1,1).tolist()

regressor = LinearRegression()
regressor.fit(X_train, y_train)
xx = np.linspace(0, 26, 100)
yy = regressor.predict(xx.reshape(xx.shape[0], 1))
plt = runplt(size=(8,8))
plt.plot(X_train, y_train, 'k.', label="train")
plt.plot(xx, yy, label="一元线性回归")

#多项式回归
quadratic_featurizer = PolynomialFeatures(degree=2)
X_train_quadratic = quadratic_featurizer.fit_transform(X_train)
X_test_quadratic = quadratic_featurizer.transform(X_test) 
regressor_quadratic = LinearRegression()

#训练数据集用来fit拟合
regressor_quadratic.fit(X_train_quadratic, y_train)
xx_quadratic = quadratic_featurizer.transform(xx.reshape(xx.shape[0], 1))
#测试数据集用来predict预测
plt.plot(xx, regressor_quadratic.predict(xx_quadratic), 'r-', label="多项式回归")
plt.legend()
plt.show()
# print(X_train)
# print(X_train_quadratic)
# print(X_test)
# print(X_test_quadratic)
# print('一元线性回归 r-squared', regressor.score(X_test, y_test))
# print('二次回归 r-squared', regressor_quadratic.score(X_test_quadratic, y_test))
