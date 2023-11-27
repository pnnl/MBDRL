import random
import numpy as np
from tensorflow.keras.layers import Input, Dense, concatenate
import tensorflow as tf
from tensorflow.keras import Model
from keras.optimizers import Adam
import os
import time
import matplotlib.pyplot as plt
import pickle

start = time.time()
X_1 = np.vstack([np.loadtxt(f'X_1_part_{i}.txt').reshape(-1, 80, 180) for i in range(3)])
Y_1 = np.vstack([np.loadtxt(f'Y_1_part_{i}.txt').reshape(-1, 80, 108) for i in range(3)])

X_1_temp = X_1
Y_1_temp = Y_1

X1 = X_1_temp[:, 12:52, :]
Y1 = Y_1_temp[:, 12:52, :]

X = X1[:,:,0:108]
U = X1[:,:,108:]
Y = Y1

Y_delta = Y - X[:,:,:108]
delta_max = Y_delta.max() # need this for scaling and re-scaling
delta_min = Y_delta.min()

X_recovery = X[:,0:35,:]

U_recovery = U
U_recovery_0 = []
U_recovery_1 = []
U_recovery_2 = []
U_recovery_3 = []
U_recovery_4 = []
for i in range(U_recovery.shape[0]):
    for j in range(U_recovery.shape[1]-5):
        #U_recovery_temp = np.concatenate((U_recovery[i, j, :], U_recovery[i, j+1, :], U_recovery[i, j+2, :], U_recovery[i, j+3, :], U_recovery[i, j+4, :]))
        U_recovery_0.append(U_recovery[i, j, :])
        U_recovery_1.append(U_recovery[i, j+1, :])
        U_recovery_2.append(U_recovery[i, j+2, :])
        U_recovery_3.append(U_recovery[i, j+3, :])
        U_recovery_4.append(U_recovery[i, j+4, :])
        
U_recovery_0 = np.reshape(U_recovery_0, (X.shape[0], 35, 72))
U_recovery_1 = np.reshape(U_recovery_1, (X.shape[0], 35, 72))
U_recovery_2 = np.reshape(U_recovery_2, (X.shape[0], 35, 72))
U_recovery_3 = np.reshape(U_recovery_3, (X.shape[0], 35, 72))
U_recovery_4 = np.reshape(U_recovery_4, (X.shape[0], 35, 72))


Y_recovery = Y
Y_recovery_5step = []
for i in range(Y_recovery.shape[0]):
    for j in range(Y_recovery.shape[1]-5):
        Y_recovery_temp = np.concatenate((Y_recovery[i, j, :], Y_recovery[i, j+1, :], Y_recovery[i, j+2, :], Y_recovery[i, j+3, :], Y_recovery[i, j+4, :]))
        Y_recovery_5step.append(Y_recovery_temp)

Y_recovery_5step = np.reshape(Y_recovery_5step, (X.shape[0], 35, 540))

# randomly select the training cases
random.seed(0)
train_cases_indx = random.sample(range(len(X_recovery)), round(len(X_recovery) * 0.9))

X_train = X_recovery[train_cases_indx, :, :]
X_test = np.delete(X_recovery, train_cases_indx, axis=0)

Y_train = Y_recovery_5step[train_cases_indx, :, :]
Y_test = np.delete(Y_recovery_5step, train_cases_indx, axis=0)

U_train_0 = U_recovery_0[train_cases_indx, :, :]
U_test_0 = np.delete(U_recovery_0, train_cases_indx, axis=0) 

U_train_1 = U_recovery_1[train_cases_indx, :, :]
U_test_1 = np.delete(U_recovery_1, train_cases_indx, axis=0) 

U_train_2 = U_recovery_2[train_cases_indx, :, :]
U_test_2 = np.delete(U_recovery_2, train_cases_indx, axis=0) 

U_train_3 = U_recovery_3[train_cases_indx, :, :]
U_test_3 = np.delete(U_recovery_3, train_cases_indx, axis=0) 

U_train_4 = U_recovery_4[train_cases_indx, :, :]
U_test_4 = np.delete(U_recovery_4, train_cases_indx, axis=0) 


# re-shape
X_train = np.reshape(X_train, (len(X_train)*35, 108))
X_test = np.reshape(X_test, (len(X_test)*35, 108))
Y_train = np.reshape(Y_train, (len(Y_train)*35, 540))
Y_test = np.reshape(Y_test, (len(Y_test)*35, 540))

U_train_0 = np.reshape(U_train_0, (len(U_train_0)*35, 72))
U_test_0 = np.reshape(U_test_0, (len(U_test_0)*35, 72))

U_train_1 = np.reshape(U_train_1, (len(U_train_1)*35, 72))
U_test_1 = np.reshape(U_test_1, (len(U_test_1)*35, 72))

U_train_2 = np.reshape(U_train_2, (len(U_train_2)*35, 72))
U_test_2 = np.reshape(U_test_2, (len(U_test_2)*35, 72))

U_train_3 = np.reshape(U_train_3, (len(U_train_3)*35, 72))
U_test_3 = np.reshape(U_test_3, (len(U_test_3)*35, 72))

U_train_4 = np.reshape(U_train_4, (len(U_train_4)*35, 72))
U_test_4 = np.reshape(U_test_4, (len(U_test_4)*35, 72))

# model structure
input_x = Input(shape=(108,))
input_u0 = Input(shape=(72,))
input_u1 = Input(shape=(72,))
input_u2 = Input(shape=(72,))
input_u3 = Input(shape=(72,))
input_u4 = Input(shape=(72,))

layer1 = Dense(1000,activation='relu')
layer2 = Dense(500, activation='relu')
layer3 = Dense(200, activation='relu')
predictions = Dense(108, activation='sigmoid')

con1 = concatenate([input_x,input_u0])
X11 = layer1(con1)
X12 = layer2(X11)
X13 = layer3(X12)
delta1 = predictions(X13)
# output = tf.slice(input_x, [0, 0], [-1, 108]) + delta_min + delta1 * (delta_max - delta_min)
output1 = input_x + delta_min + delta1 * (delta_max - delta_min)

con2 = concatenate([output1,input_u1])
X21 = layer1(con2)
X22 = layer2(X21)
X23 = layer3(X22)
delta2 = predictions(X23)
output2 = output1 + delta_min + delta2 * (delta_max - delta_min)

con3 = concatenate([output2,input_u2])
X31 = layer1(con3)
X32 = layer2(X31)
X33 = layer3(X32)
delta3 = predictions(X33)
output3 = output2 + delta_min + delta3 * (delta_max - delta_min)

con4 = concatenate([output3,input_u3])
X41 = layer1(con4)
X42 = layer2(X41)
X43 = layer3(X42)
delta4 = predictions(X43)
output4 = output3 + delta_min + delta4 * (delta_max - delta_min)

con5 = concatenate([output4,input_u4])
X51 = layer1(con5)
X52 = layer2(X51)
X53 = layer3(X52)
delta5 = predictions(X53)
output5 = output4 + delta_min + delta5 * (delta_max - delta_min)
output = tf.concat([output1,output2,output3,output4,output5], axis=1)
model = Model(inputs=[input_x,input_u0,input_u1,input_u2,input_u3,input_u4], outputs=output)

model.summary()

model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.001), metrics='mean_squared_error')

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15)

history = model.fit([X_train,U_train_0,U_train_1,U_train_2,U_train_3,U_train_4], Y_train, epochs=100, batch_size=16, validation_split=0.1, callbacks=[callback], verbose = 2)
end = time.time()
print(end - start)

save_path ='trained_surrogate_model.h5'
model.save(save_path)

# check the MSE and loss plot to make sure training is finished
plt.plot(history.history['mean_squared_error'])
plt.plot(history.history['val_mean_squared_error'])
#plt.ylim(0, 0.0001)
plt.title('Model MSE')
plt.ylabel('MSE')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.tight_layout()
plt.savefig('MSE.png')
plt.close()


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
#plt.ylim(0, 0.0001)
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.tight_layout()
plt.savefig('loss.png')
plt.close()

with open('trainHistoryDict', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)