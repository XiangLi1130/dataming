
# coding: utf-8

# In[2]:


import sklearn
import numpy as np
import random
from sklearn.datasets import fetch_rcv1
import math
from sklearn import svm
from scipy import sparse
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding
from keras.layers import LSTM
from keras.optimizers import SGD

# In[3]:


rcv1 = fetch_rcv1()


# In[4]:


rcv1.data.shape


# In[5]:


def Q1_label():
    label = [-1] * 804414
    for i in range (0,804414):
        if(33 in rcv1['target'][i].indices):
            label[i] = 1
    return label
label = Q1_label()
print('generate label: ')
print(label)


# In[6]:


def Q1_extract_data():
    train_data = rcv1['data'][0:100000]
    test_data =  rcv1['data'][100000:]
    train_target = rcv1['target'][0:100000]
    test_target = rcv1['target'][100000:804414]
    return train_data,test_data,train_target,test_target
train_data,test_data,train_target,test_target = Q1_extract_data()


# In[7]:


def generate_At(X,Y,size):
    index = random.sample(range(X.shape[0]), size)
    return X[index], Y[index]


# In[35]:


def generate_AtPlus(Atx, Aty,wt):
    res = np.array(Atx.toarray() .dot( np.transpose(wt))) * np.array(Aty)
    validIndex = (np.array(Atx.toarray().dot(np.transpose(wt))) * np.array(Aty) < 0.05)
    return Atx[validIndex[:, 0]], np.transpose(Aty[validIndex])


# In[24]:


def get_Delta(Atx,Aty,Atx_,Aty_,lamda,wt):
    x = Atx_
    y = Aty_
    res = x.multiply(sparse.csr_matrix(y)).sum(axis = 0)
    return lamda * wt  - 1 / Atx.shape[0] * res


# In[10]:


def calculate_trainingError(wt,X,Y,rate):
    predictY = X*np.transpose(wt)
    # if sign of predictY is different from the exact label, there is a training error 
    tempY = Y
    sign = np.array(predictY) * np.array(tempY)
    error_rate = (sign <= 0).sum()/X.shape[0]
    rate.append(error_rate)


# In[28]:


def Q2_pegasos(X,Y,T,lamda,size):
    rate = []
    wt = (np.random.rand(1, X.shape[1]) / math.sqrt(X.shape[1] * lamda))
    for i in range(1, T + 1):
        Atx, Aty = generate_At(X,Y,size)
        Atx_, Aty_ = generate_AtPlus(Atx, Aty,wt) 
        DeltaT = get_Delta(Atx,Aty,Atx_,Aty_,lamda,wt)
        etaT = 1/(i * lamda)
        wt_ = wt - etaT * DeltaT
        wt = min(1,(1/math.sqrt(lamda))/np.linalg.norm(wt_))*wt_
        calculate_trainingError(wt,X,Y,rate)
    return wt,rate,T


# In[12]:


def Q2_plot(rate,T):
    x_axis = range(1,T+1)
    plt.plot(x_axis,rate)
    plt.show()


# In[54]:


def Q2_plot2(rate1, rate2 ,T):
    x_axis = range(1,T+1)
    plt.plot(x_axis,rate1,label = 'pegasos')
    plt.plot(x_axis,rate2, label = 'Adagrad')
    plt.legend();
    plt.show()


# In[38]:


batchSizeList = [10,20,50,100,300]
lamdaList = [0.01,0.05,0.1,0.15,1,1.05]


# In[39]:


def Q2_tuneParameter(X,Y,T,lamdaList,batchSizeList):
    rateListBatch = []
    rateListLamda = []
    lamda = 0.01
    wt = (np.random.rand(1, X.shape[1]) / math.sqrt(X.shape[1] * lamda))
    for a in range(len(batchSizeList)):
        rate = []
        for i in range(1, T + 1):
            Atx, Aty = generate_At(X,Y,batchSizeList[a])
            Atx_, Aty_ = generate_AtPlus(Atx, Aty,wt) 
            DeltaT = get_Delta(Atx,Aty,Atx_,Aty_,lamda,wt)
            etaT = 1/(i * lamda)
            wt_ = wt - etaT * DeltaT
            wt = min(1,(1/math.sqrt(lamda))/np.linalg.norm(wt_))*wt_
            calculate_trainingError(wt,X,Y,rate)
        rateListBatch.append(rate)
    for b in range(len(lamdaList)):
        rate = []
        for i in range(1, T + 1):
            Atx, Aty = generate_At(X,Y,50)
            Atx_, Aty_ = generate_AtPlus(Atx, Aty,wt) 
            DeltaT = get_Delta(Atx,Aty,Atx_,Aty_,lamdaList[b],wt)
            etaT = 1/(i * lamdaList[b])
            wt_ = wt - etaT * DeltaT
            wt = min(1,(1/math.sqrt(lamdaList[b]))/np.linalg.norm(wt_))*wt_
            calculate_trainingError(wt,X,Y,rate)
        rateListLamda.append(rate)
    return T, lamdaList,batchSizeList, rateListBatch,rateListLamda


# In[40]:


Y = np.transpose(np.matrix(label[0:100000]))
X = train_data


# In[41]:


def parameter_plot(labelList, dataList, T):
    x_axis = range(1,T+1)
    for i in range(len(labelList)):
        plt.plot(x_axis,dataList[i],label = str(labelList[i]))
    plt.legend();
    plt.show()


# In[42]:


T, lamdaList,batchSizeList, rateListBatch,rateListLamda = Q2_tuneParameter(X,Y,T,lamdaList,batchSizeList)
parameter_plot(batchSizeList, rateListBatch, T)
parameter_plot(lamdaList,rateListLamda,T)


# In[36]:


#error rate on train data
Y = np.transpose(np.matrix(label[0:100000]))
X = train_data
wt,rate,T = (Q2_pegasos(X,Y,1000,0.01,100))
Q2_plot(rate,T)


# In[37]:


#train error for pegasos
print(rate[len(rate)-1])


# In[43]:


test_y = np.transpose(np.matrix(label[100000:804414]))
test_x = test_data


# In[44]:


def Q2_pegasos_test(X,Y,T,lamda,size,test_x,test_y):
    rate = []
    wt = (np.random.rand(1, X.shape[1]) / math.sqrt(X.shape[1] * lamda))
    for i in range(1, T + 1):
        Atx, Aty = generate_At(X,Y,size)
        Atx_, Aty_ = generate_AtPlus(Atx, Aty,wt) 
        DeltaT = get_Delta(Atx,Aty,Atx_,Aty_,lamda,wt)
        etaT = 1/(i * lamda)
        wt_ = wt - etaT * DeltaT
        wt = min(1,(1/math.sqrt(lamda))/np.linalg.norm(wt_))*wt_
        calculate_trainingError(wt,test_x,test_y,rate)
    return wt,rate,T


# In[50]:


#error rate on test data
wt,rate2,T = (Q2_pegasos_test(X,Y,1000,0.01,100,test_x,test_y))
# print(rate2)
Q2_plot2(rate, rate2, T)


# In[46]:


Q2_plot(rate2, T)
#test error for pegasos
print(rate2[len(rate2)-1])


# In[51]:


def adagrad(learning_rate, X, Y, T ,batchSize):
    w = [0] *  X.shape[1]
    s = [1] * X.shape[1]
    s = np.array(s)
    multiMatrix = X.multiply(np.reshape(Y, (X.shape[0], 1))).tocsr()
    rate = []
    for i in range(T):
        for j in range(batchSize):
            index = random.randint(0, X.shape[0]-1)
            y_hat = multiMatrix.getrow(index).dot(np.transpose(w))
            if (y_hat < 1):
                yi = Y[index][0]
                delta = -X.getrow(index).multiply(np.array(yi))
                s = np.add(s, np.square(delta.todense()))
                w = w - learning_rate * delta / (np.sqrt(s))
        calculate_trainingError(w,X,Y,rate)
    return rate, T, w


# In[263]:


Y = np.transpose(np.matrix(label[0:100000]))
X = train_data
rate1, T,w = adagrad(1/math.sqrt(1000), X, Y, 1000,50)
rate2, T,w = adagrad(1/math.sqrt(1000), X, Y, 1000,100)
rate3, T,w = adagrad(1/math.sqrt(1000), X, Y, 1000,200)
rate4, T,w = adagrad(1/math.sqrt(1000), X, Y, 1000,500)
labelList = [50,100,200,500]
dataList = [rate1,rate2,rate3,rate4]
parameter_plot(labelList, dataList, T)
# Q2_plot(rate1,T)


# In[52]:


Y = np.transpose(np.matrix(label[0:100000]))
X = train_data
rate4, T,w = adagrad(1/math.sqrt(1000), X, Y, 1000,500)
Q2_plot(rate4,1000)


# In[55]:


Q2_plot2(rate, rate4,1000)


# In[264]:


#train error for adagrad
print(rate4[999])


# In[266]:


#predict on test data
rate, T, w = adagrad(1/math.sqrt(1000), X, Y, 1000,500)
test_x = test_data
test_y = np.transpose(np.matrix(label[100000:]))
calculate_trainingError(w,test_x,test_y,rate)


# In[268]:


#test error for adagrad
print('test error rate is: ', rate[len(rate)-1])




model = Sequential()

model.add(Dense(units=100, activation='relu', input_dim=47236))
model.add(Dense(units=100, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
model.fit(x_train, label_train, epochs=5, batch_size=100)
model.summary()
accuracy = model.evaluate(x_train, label_train, batch_size=100)
print('1 hidden layer Train accuracy:', accuracy)


# In[106]:


# 2 hidden layer
model = Sequential()

model.add(Dense(units=100, activation='relu', input_dim=47236))
model.add(Dense(units=100, activation='relu'))
model.add(Dense(units=100, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
model.fit(x_train, label_train, epochs=5, batch_size=100)
model.summary()
accuracy = model.evaluate(x_train, label_train, batch_size=100)
print('2 hidden layer Train accuracy:', accuracy)


# In[107]:


# 3 hidden layer
model = Sequential()

model.add(Dense(units=100, activation='relu', input_dim=47236))
model.add(Dense(units=100, activation='relu'))
model.add(Dense(units=100, activation='relu'))
model.add(Dense(units=100, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
model.fit(x_train, label_train, epochs=5, batch_size=100)
model.summary()
accuracy = model.evaluate(x_train, label_train, batch_size=100)
print('3 hidden layer Train accuracy:', accuracy)


# In[9]:


x = [1, 2, 3]
y = [(1 - 0.9079600015282631) *100, (1 - 0.924530002117157) *100, (1 - 0.936050002336502)*100]
plt_name = 'Training Error with Different Hidden Layer'
plt.title(plt_name)
plt.plot(x,y)
plt.xticks(range(1,len(x) +1))
plt.xlabel('Hidden Layer')
plt.ylabel('Training Error %')
plt.savefig(plt_name+".png")
plt.clf()


# # 4b

# In[10]:


# no hidden layer
model = Sequential()

model.add(Dense(units=100, activation='relu', input_dim=47236))
model.add(Dense(units=1, activation='sigmoid'))
# sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
model.fit(x_train, label_train, epochs=5, batch_size=100)
#classes = model.predict(x_test, batch_size=128)
model.summary()
accuracy = model.evaluate(x_train, label_train, batch_size=100)
print('3 hidden layer Train accuracy:', accuracy)



# In[11]:


# 1 hidden layer
model = Sequential()

model.add(Dense(units=100, activation='relu', input_dim=47236))
model.add(Dense(units=100, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))
# sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
model.fit(x_train, label_train, epochs=5, batch_size=100)

model.summary()
accuracy = model.evaluate(x_train, label_train, batch_size=100)
print('3 hidden layer Train accuracy:', accuracy)


# In[12]:


# 2 hidden layer
model = Sequential()
model.add(Dense(units=100, activation='relu', input_dim=47236))
model.add(Dense(units=100, activation='relu'))
model.add(Dense(units=100, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))
# sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
model.fit(x_train, label_train, epochs=5, batch_size=100)
model.summary()
# score = model.evaluate(x_test, label_test, batch_size=100)
# print('Test score:', score)
accuracy = model.evaluate(x_train, label_train, batch_size=100)
print('3 hidden layer Train accuracy:', accuracy)


# In[13]:


# 3 hidden layer
model = Sequential()
model.add(Dense(units=100, activation='relu', input_dim=47236))
model.add(Dense(units=100, activation='relu'))
model.add(Dense(units=100, activation='relu'))
model.add(Dense(units=100, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))
# sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
model.fit(x_train, label_train, epochs=5, batch_size=100)

model.summary()

# score = model.evaluate(x_test, label_test, batch_size=100)
# print('Test score:', score)
accuracy = model.evaluate(x_train, label_train, batch_size=100)
print('3 hidden layer Train accuracy:', accuracy)


# In[14]:


# 4 hidden layer
model = Sequential()
model.add(Dense(units=100, activation='relu', input_dim=47236))
model.add(Dense(units=100, activation='relu'))
model.add(Dense(units=100, activation='relu'))
model.add(Dense(units=100, activation='relu'))
model.add(Dense(units=100, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))
# sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
model.fit(x_train, label_train, epochs=5, batch_size=100)

model.summary()

# score = model.evaluate(x_test, label_test, batch_size=100)
# print('Test score:', score)
accuracy = model.evaluate(x_train, label_train, batch_size=100)
print('3 hidden layer Train accuracy:', accuracy)


# In[15]:


# 5 hidden layer
model = Sequential()
model.add(Dense(units=100, activation='relu', input_dim=47236))
model.add(Dense(units=100, activation='relu'))
model.add(Dense(units=100, activation='relu'))
model.add(Dense(units=100, activation='relu'))
model.add(Dense(units=100, activation='relu'))
model.add(Dense(units=100, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))
# sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
model.fit(x_train, label_train, epochs=5, batch_size=100)

model.summary()

# score = model.evaluate(x_test, label_test, batch_size=100)
# print('Test score:', score)
accuracy = model.evaluate(x_train, label_train, batch_size=100)
print('3 hidden layer Train accuracy:', accuracy)


# In[16]:


# 6 hidden layer
model = Sequential()
model.add(Dense(units=100, activation='relu', input_dim=47236))
model.add(Dense(units=100, activation='relu'))
model.add(Dense(units=100, activation='relu'))
model.add(Dense(units=100, activation='relu'))
model.add(Dense(units=100, activation='relu'))
model.add(Dense(units=100, activation='relu'))
model.add(Dense(units=100, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))
# sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
model.fit(x_train, label_train, epochs=5, batch_size=100)

model.summary()

# score = model.evaluate(x_test, label_test, batch_size=100)
# print('Test score:', score)
accuracy = model.evaluate(x_train, label_train, batch_size=100)
print('3 hidden layer Train accuracy:', accuracy)


# In[23]:


# Test error for different hidden layer
import matplotlib.pyplot as plt
x = [0, 1, 2, 3, 4, 5, 6]
y = [(1 - 0.8168903510672614) *100, (1 - 0.907185264844609) *100, (1 - 0.9237124783598789)*100, (1 - 0.9351872648048789)*100, (1 - 0.9377070900480086)*100, (1 - 0.9362888899691572)*100, (1 - 0.9242363182345853)*100]
plt_name = 'Test Error with Different Hidden Layer (Hidden unites = 100)'
plt.title(plt_name)
plt.plot(x,y)
plt.xticks(range(1,len(x) +1))
plt.xlabel('Hidden Layer')
plt.ylabel('Test Error %')
plt.savefig(plt_name+".png")
plt.clf()


# In[22]:


# tarining error for different hidden layer

x = [0, 1, 2, 3, 4, 5, 6]
y = [(1 - 0.8361699987649918) *100, (1 - 0.9060200013518334) *100, (1 - 0.9276800018548965)*100, (1 - 0.9331700021624565)*100, (1 - 0.9469900039434433)*100, (1 - 0.9471100035309792)*100, (1-0.9496600027084351)*100]
plt_name = 'Training Error with Different Hidden Layer (Hidden unites = 100)'
plt.title(plt_name)
plt.plot(x,y)
plt.xticks(range(1,len(x) +1))
plt.xlabel('Hidden Layer')
plt.ylabel('Training Error %')
plt.savefig(plt_name+".png")
plt.clf()



# 4 hidden layer 
model = Sequential()
model.add(Dense(units=50, activation='relu', input_dim=47236))
model.add(Dense(units=50, activation='relu'))
model.add(Dense(units=50, activation='relu'))
model.add(Dense(units=50, activation='relu'))
model.add(Dense(units=50, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))
# sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
model.fit(x_train, label_train, epochs=5, batch_size=100)

model.summary()

# score = model.evaluate(x_test, label_test, batch_size=100)
# print('Test score:', score)
accuracy = model.evaluate(x_train, label_train, batch_size=100)
print('4 hidden layer Train accuracy:', accuracy)


# In[25]:


# 4 hidden layer 
model = Sequential()
model.add(Dense(units=75, activation='relu', input_dim=47236))
model.add(Dense(units=75, activation='relu'))
model.add(Dense(units=75, activation='relu'))
model.add(Dense(units=75, activation='relu'))
model.add(Dense(units=75, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))
# sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
model.fit(x_train, label_train, epochs=5, batch_size=100)

model.summary()

# score = model.evaluate(x_test, label_test, batch_size=100)
# print('Test score:', score)
accuracy = model.evaluate(x_train, label_train, batch_size=100)
print('4 hidden layer Train accuracy:', accuracy)


# In[26]:


# 4 hidden layer 
model = Sequential()
model.add(Dense(units=25, activation='relu', input_dim=47236))
model.add(Dense(units=25, activation='relu'))
model.add(Dense(units=25, activation='relu'))
model.add(Dense(units=25, activation='relu'))
model.add(Dense(units=25, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
model.fit(x_train, label_train, epochs=5, batch_size=100)

model.summary()

# score = model.evaluate(x_test, label_test, batch_size=100)
# print('Test score:', score)
accuracy = model.evaluate(x_train, label_train, batch_size=100)
print('4 hidden layer Train accuracy:', accuracy)



# 6.2
x = [25, 50, 75, 100]
y = [(1 - 0.9381000028252602) *100, (1 - 0.9450800023674965) *100, (1 - 0.91934000146389)*100, (1 - 0.9469900039434433)*100]
plt_name = 'Training Error with Different Hidden Unites (Hidden Layer = 4)'
plt.title(plt_name)
plt.plot(x,y)
plt.xlabel('Hidden Layer')
plt.ylabel('Training Error %')

plt.savefig(plt_name+".png")
plt.clf()
