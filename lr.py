from six.moves import cPickle as pickle
from sklearn import linear_model
import numpy as np

pickle_file = open('notMNIST.pickle','rb')
data = pickle.load(pickle_file)
train_dataset = data['train_dataset']
train_labels = data['train_labels']
valid_dataset = data['valid_dataset']
test_dataset = data['test_dataset']
test_labels = data['test_labels']


new_train_dataset= np.reshape(train_dataset[:,:,:],(train_dataset.shape[0],28*28))
#regr = linear_model.LinearRegression()
#regr = linear_model.LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='lbfgs', max_iter=100, multi_class='multinomial', verbose=0, warm_start=False, n_jobs=1)
regr = linear_model.LogisticRegression(multi_class='multinomial', solver='lbfgs', penalty='l2')

regr.fit(new_train_dataset, train_labels[:])
print("Residual sum of squares: %.2f"
      % np.mean((regr.predict(np.reshape(test_dataset[0:100],(100,28*28))) - test_labels[0:100]) ** 2))
round_result = map(lambda x:round(x), regr.predict(np.reshape(test_dataset[0:100],(100,28*28))))
diff =round_result - test_labels[0:100]
print(round_result)
print("\n")
print(diff)
print("\n")
print(test_labels[0:100])

matching_result = filter(lambda x : x==0, diff)
print(len(matching_result))
#print(valid_dataset.shape)