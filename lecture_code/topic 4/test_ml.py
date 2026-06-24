import numpy as np
# from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

np.random.seed(2)

x = np.random.normal(loc=3, scale=1, size=1000)
y = np.random.normal(150, 40, 1000) / x

# plt.scatter(x, y)
# plt.show()

def mse(u, v):
	return .5 * np.mean((u-v)**2)
	
train_x, valid_x, test_x = x[:600], x[600:800], x[800:]
train_y, valid_y, test_y = y[:600], y[600:800], y[800:]

myline = np.linspace(start=0, stop=6, num=1000)
ERR_VALID = []
# hyperparameter deg
for deg in range(1, 10):
	mymodel = np.poly1d(np.polyfit(train_x, train_y, deg=deg))

	ERR_VALID.append(r2_score(valid_y, mymodel(valid_x)))

print([f'{x:.3f}' for x in ERR_VALID])

deg_opt = np.argmin(ERR_VALID)
print(f'The optimal degree is {deg_opt + 1}')




# plt.scatter(train_x, train_y)
# plt.plot(myline, mymodel(myline))
# plt.show()

# r2 = r2_score(train_y, mymodel(train_x))
# print(r2)

# r2 = r2_score(test_y, mymodel(test_x))
# print(r2)

