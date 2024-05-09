import pandas as pd
import numpy as np
import random as rand
import sympy as sp
from sklearn.preprocessing import StandardScaler

def loss_function(y_true, y_pred):
    diff = y_true - y_pred
    diff_transpose = np.transpose(diff)
    return np.dot(diff_transpose, diff)

def find_gradient(X, y, coefficient_list):
    y_predicted = X.dot(coefficient_list)
    residuals = y - y_predicted
    gradient = -2 * X.T.dot(residuals)
    return gradient

auto_mpg = pd.read_fwf('auto-mpg.data', header=None)
auto_mpg.drop(auto_mpg.columns[8], axis = 1, inplace = True)

print(max(auto_mpg.iloc[:,0]))
# scaler = MinMaxScaler()
scaler = StandardScaler()
# auto_mpg = scaler.fit_transform(auto_mpg)

# auto_mpg = np.array(auto_mpg)
y = auto_mpg.iloc[:,0]
X = auto_mpg.iloc[:,1:8]
X = scaler.fit_transform(X)
y = scaler.fit_transform(y.to_frame())

y_train = y[:353, 0]
y_test = y[353:392, 0]
# auto_mpg = np.delete(auto_mpg, 0, axis=1)




# X_train = np.transpose(np.array(X_train))
X = np.c_[np.ones(X.shape[0]), X]
X_test = X[353:392, :]
X_train = X[0:353, :]

coeff_list = np.array([rand.randint(12,35) for i in range(8)])
# intercept, coef_1, coef_2, coef_3, coef_4, coef_5, coef_6, coef_7 = sp.symbols(u"β_0, β_1, β_2, β_3, β_4, β_5, β_6, β_7")
# coeff_list = [intercept, coef_1, coef_2, coef_3, coef_4, coef_5, coef_6, coef_7]
# coeff_list = np.array(coeff_list)
y_predicted = np.dot(X_train, coeff_list)
loss_function_eq = loss_function(y_train, y_predicted)
diff = y_train - y_predicted
rss = -2*X_train.T@diff
coeff_list = rss
# print(loss_function_eq)
# print(rss)
#
# y_predicted = np.dot(X_train, coeff_list)
# loss_function_eq2 = loss_function(y_train, y_predicted)
# diff = y_train - y_predicted
# rss = -2*X_train.T@diff
#
# print(loss_function_eq2)
# print(rss)
#
# print(abs(loss_function_eq - loss_function_eq2))
alpha = .000009



# while loss_function_eq == 100:
#     for i in range(len(coeff_list)):
#         coeff_list[i] = coeff_list[i] - alpha * rss[i]
#     y_predicted = np.dot(X_train, coeff_list)
#     loss_function_eq = loss_function(y_train, y_predicted)
#     X_train = np.transpose(X_train)
#     diff = y_train - y_predicted
#     rss = -2 * X_train.T @ diff
#     print("WE IN HERE")

for iteration in range(100000):
    rss_gradient = find_gradient(X_train, y_train, coeff_list)
    coeff_list = coeff_list - alpha * rss_gradient

    y_predicted = np.dot(X_train, coeff_list)
    loss = loss_function(y_train, y_predicted)

    if abs(loss - loss_function_eq) < 1e-3:
        break

    loss = loss_function_eq

print(coeff_list.shape)
print("Found optimal parameters:", coeff_list)
test_result = np.dot(X_test, coeff_list)
# # print(test_result, y_test[37])
# print(test_result.shape, y_test.shape)
# for (x, y) in zip(test_result, y_test):
#     print(abs(x - y))

def cost(X, y, b,n):
    # return (np.sum((np.dot(X, b) - np.array(y))**2)/n)**0.5
    return ((np.sum((np.dot(X, b) - np.array(y))**2)))**0.5

rmse = cost(X_train, y_train, coeff_list, X_train.shape[0])
print("RMSE:", rmse)

y_predicted = X_test.dot(coeff_list)
print(y_test.shape)
print(y_predicted.shape)
# scaler.inverse_transform(y_test)
# y_predicted = y_predicted.reshape(1,-1)
# print(scaler.inverse_transform(y_predicted))