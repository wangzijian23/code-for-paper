import numpy as np
import pandas
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor
import random
from sklearn.metrics import r2_score

data = pandas.read_excel('D:/王子健\大三/遥感原理与方法二/@gl/处理1/输出excel/宋各庄/归一化结果.xlsx')
X = data.iloc[:, 0:7]
Y = data.iloc[:, 7]
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)  # 自动生成数据
regr = ExtraTreesRegressor(n_estimators=1600, min_samples_split=5, min_samples_leaf=1, bootstrap=False, max_depth= 66)
regr = regr.fit(x_train, y_train)   # 训练训练数据集
Y_test_p = regr.predict(x_test)
feature_importances_ = regr.feature_importances_  # 重要程度

y_pred = Y_test_p * -8974549.05364298
y_true = y_test.iloc[:].values * -8974549.05364298
RMSE = np.linalg.norm(y_pred-y_true, ord=2)/len(y_pred)**0.5
print(r2_score(y_true, y_pred, multioutput='raw_values'))
print(RMSE)
print(feature_importances_)

x1 = data.iloc[:, 0]
x2 = data.iloc[:, 1]
x3 = data.iloc[:, 2]
x4 = data.iloc[:, 3]
x5 = data.iloc[:, 4]
x6 = data.iloc[:, 5]
x7 = data.iloc[:, 6]

pred = np.zeros((12500, 7))
for i in range(0, 12500):
    a1 = random.choice(x1)
    a2 = random.choice(x2)
    a3 = random.choice(x3)
    a4 = random.choice(x4)
    a5 = random.choice(x5)
    a6 = random.choice(x6)
    a7 = random.choice(x7)

    pred[i][0] = a1
    pred[i][1] = a2
    pred[i][2] = a3
    pred[i][3] = a3
    pred[i][4] = a5
    pred[i][5] = a6
    pred[i][6] = a7

mtklpred = regr.predict(pred)
mtklpred = mtklpred * -8974549.05364298
np.savetxt("1mtkl.xlsx", mtklpred)
