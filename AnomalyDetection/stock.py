import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Model, load_model, Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import ModelCheckpoint, TensorBoard
import seaborn as sns
from sklearn.metrics import (confusion_matrix, precision_recall_curve, auc,
                             roc_curve, recall_score, classification_report, f1_score,
                             precision_recall_fscore_support)

df = pd.read_csv("data/AABA_2006-01-01_to_2018-01-01.csv")

df = df.drop(['Open', 'High', 'Low', 'Volume', 'Name'], axis=1)
df['Close'] = MinMaxScaler().fit_transform(df['Close'].values.reshape(-1, 1))
print(df.shape)

train = df[df.Date < '2015-01-01']
train = train.drop(['Date'], axis=1)
test = df[df.Date >= '2015-01-01']
test = test.drop(['Date'], axis=1)
print(test.shape)

for i in range(1, 13):
    train['shift_{}'.format(i)] = train['Close'].shift(i)
    test['shift_{}'.format(i)] = test['Close'].shift(i)

X_train = train.dropna().drop('Close', axis=1)
y_train = train.dropna()[['Close']]
X_test = test.dropna().drop('Close', axis=1)
y_test = test.dropna()[['Close']]
X_train = X_train.values
y_train = y_train.values
X_test = X_test.values
y_test = y_test.values
X_train = X_train.reshape(X_train.shape[0], 12, 1)
X_test = X_test.reshape(X_test.shape[0], 12, 1)



model = Sequential()
model.add(LSTM(20, input_shape=(12, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()



nb_epoch = 15
batch_size = 64

checkpointer = ModelCheckpoint(filepath="stock_model.h5",
                               verbose=0,
                               save_best_only=True)
tensorboard = TensorBoard(log_dir='./logs',
                          histogram_freq=0,
                          write_graph=True,
                          write_images=True)
history = model.fit(X_train, y_train,
                    epochs=nb_epoch,
                    batch_size=batch_size,
                    shuffle=True,
                    validation_data=(X_test, y_test),
                    verbose=1,
                    callbacks=[checkpointer, tensorboard]).history


plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()


model = load_model('stock_model.h5')


predictions = model.predict(X_test)

plt.plot(y_test)
plt.plot(predictions)
plt.title('prediction')
plt.ylabel('price')
plt.xlabel('time')
plt.legend(['y', 'prediction'], loc='upper right')
plt.show()

