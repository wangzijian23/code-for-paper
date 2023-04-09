import numpy as np
import pandas
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import RandomizedSearchCV

EXT = ExtraTreesRegressor()
data = pandas.read_excel('D:\王子健\大三\遥感原理与方法二\@gl\处理1\输出excel\宋各庄\归一化结果.xlsx')
X = data.iloc[:, 0:7]
Y = data.iloc[:, 7]
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)  # 自动生成数据

n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=19)]
min_samples_split = [2, 5, 10, 20]
min_samples_leaf = [1, 2, 4]
bootstrap = [True,False]

random_params_group = {'n_estimators':n_estimators,
                      'min_samples_split':min_samples_split,
                      'min_samples_leaf':min_samples_leaf,
                      'bootstrap':bootstrap}

random_model = RandomizedSearchCV(EXT,param_distributions = random_params_group,n_iter = 100,
scoring = 'neg_mean_squared_error',verbose = 2,n_jobs = 1,cv = 3,random_state = 0)
random_model.fit(x_train, y_train)
print(random_model.best_params_)