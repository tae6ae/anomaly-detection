import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.metrics import (confusion_matrix, precision_recall_curve, auc,
                             roc_curve, recall_score, classification_report, f1_score,
                             precision_recall_fscore_support)

df_train = pd.read_csv("data/aps_failure_training_set_processed_8bit.csv")
df_test = pd.read_csv("data/aps_failure_test_set_processed_8bit.csv")

df_train = df_train.rename(columns = {'class': 'Class'})
df_test = df_test.rename(columns = {'class': 'Class'})

df_train.loc[df_train.Class < 0, 'Class'] = 0
df_train.loc[df_train.Class > 0, 'Class'] = 1

df_test.loc[df_test.Class < 0, 'Class'] = 0
df_test.loc[df_test.Class > 0, 'Class'] = 1


# features = df_train.columns.tolist()
'''
features = ['Class', 'aa_000', 'ag_005', 'ah_000', 'ap_000', 'aq_000', 'az_005', 'ba_000', 
            'ba_001', 'ba_002', 'ba_003', 'ba_004', 'ba_005', 'ba_006', 'bb_000', 'bg_000', 
            'bi_000', 'bj_000', 'bt_000', 'bu_000', 'bv_000', 'cc_000', 'ci_000', 'ck_000', 
            'cn_004', 'cn_005', 'dn_000', 'ds_000', 'ee_000', 'ee_001', 'ee_004']

features = ['Class', 'bm_000', 'br_000', 'bp_000', 'bq_000', 'bo_000', 'bn_000']
'''
features = ['Class', 'bm_000', 'br_000', 'bp_000', 'bq_000', 'bo_000', 'bn_000', 'aa_000', 'ba_003', 'ee_000', 'ee_001']

f = features[1:-1]
df_train[f] = StandardScaler().fit_transform(df_train[f])
df_test[f] = StandardScaler().fit_transform(df_test[f])


X_train = df_train[df_train.Class == 0]
X_train = X_train.drop(['Class'], axis=1)
y_test = df_test['Class']
X_test = df_test.drop(['Class'], axis=1)

# '''
input_dim = X_train.shape[1]
encoding_dim = 7

input_layer = Input(shape=(input_dim, ))
encoder = Dense(encoding_dim, activation="tanh")(input_layer)
encoder = Dense(int(encoding_dim / 2), activation="tanh")(encoder)
decoder = Dense(int(encoding_dim), activation='tanh')(encoder)
decoder = Dense(input_dim, activation='tanh')(decoder)
autoencoder = Model(inputs=input_layer, outputs=decoder)



nb_epoch = 150
batch_size = 64
autoencoder.compile(optimizer='adam',
                    loss='mean_squared_error',
                    metrics=['accuracy'])
checkpointer = ModelCheckpoint(filepath="aps_model.h5",
                               verbose=0,
                               save_best_only=True)
tensorboard = TensorBoard(log_dir='./logs',
                          histogram_freq=0,
                          write_graph=True,
                          write_images=True)
history = autoencoder.fit(X_train, X_train,
                    epochs=nb_epoch,
                    batch_size=batch_size,
                    shuffle=True,
                    validation_data=(X_test, X_test),
                    verbose=1,
                    callbacks=[checkpointer, tensorboard]).history


plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()
# '''

autoencoder = load_model('aps_model.h5')


predictions = autoencoder.predict(X_test)
mse = np.mean(np.power(X_test - predictions, 2), axis=1)
error_df = pd.DataFrame({'reconstruction_error': mse,
                        'true_class': y_test})


fpr, tpr, thresholds = roc_curve(error_df.true_class, error_df.reconstruction_error)
roc_auc = auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, label='AUC = %0.4f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.001, 1])
plt.ylim([0, 1.001])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')



threshold = 0.2

groups = error_df.groupby('true_class')
fig, ax = plt.subplots()

for name, group in groups:
    ax.plot(group.index, group.reconstruction_error, marker='o', ms=3.5, linestyle='',
            label= "neg" if name == 1 else "pos")
ax.hlines(threshold, ax.get_xlim()[0], ax.get_xlim()[1], colors="r", zorder=100, label='Threshold')
ax.legend()
plt.title("Reconstruction error for different classes")
plt.ylabel("Reconstruction error")
plt.xlabel("Data point index")
# plt.show()


LABELS = ["neg", "pos"]

y_pred = [1 if e > threshold else 0 for e in error_df.reconstruction_error.values]
conf_matrix = confusion_matrix(error_df.true_class, y_pred)
plt.figure(figsize=(6, 6))
sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
plt.title("Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()

'''
for i in range(0, 10):
    threshold = 1.5 + (i / 10)
    y_pred = [1 if e > threshold else 0 for e in error_df.reconstruction_error.values]
    conf_matrix = confusion_matrix(error_df.true_class, y_pred)
    tn = conf_matrix[0][0]
    fp = conf_matrix[0][1]
    fn = conf_matrix[1][0]
    tp = conf_matrix[1][1]
    acc = (tp + tn) / (tp + tn + fp + fn)
    pre = tp / (tp + fp)
    rec = tp / (tp + fn)
    print('threshold :', round(threshold, 4))
    print('accuracy :', round(acc, 4))
    print('precision :', round(pre, 4))
    print('recall :', round(rec, 4))
    print('f1 score :', round(f1_score(y_pred, y_test), 4))
    print('\n')
'''
