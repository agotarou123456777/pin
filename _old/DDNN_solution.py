
import tensorflow as tf
import keras.losses as losses
import keras.optimizers as optimizers
from keras.layers import Dense
from keras.models import Sequential
from keras.initializers import GlorotNormal
from keras.callbacks import EarlyStopping

import os
import matplotlib.pyplot as plt
import numpy as np

from lib import lib_FDM as fdm
from lib import lib_NNsim as nsim




def solve_DDNNs():
    '''
    STEP 1
    解析解を求める
    '''
    gamma = 2 # ダンパーの減衰係数
    omega = 20 # 固有角周波数

    t = np.linspace(0,1,500) # タイムステップを生成(0s-1sの間を500分割)
    t = np.reshape(t,[-1,1])
    x = fdm.analytical_solution(gamma, omega, t) # タイムステップを用いて解析解を計算
    x = np.reshape(x,[-1,1])

    '''
    # プロット作成
    plt.plot(t, x, color='darkorange', label='Analytic Solution Result')
    plt.title('Analytic Solution Result')
    plt.xlabel("time")
    plt.ylabel("displacement")
    plt.legend()
    plt.show()
    '''
    # 出力されたグラフ上で'q'を押すと次のプロットへ
    print("Please press Q-key")

    '''
    STEP 2
    解析解から学習用のデータポイントを作成
    '''
    # Data points
    datapoint_list = [i for i in range(0,300,10)] # 学習用データとして抜き出す箇所を設定(0-300要素の間を等間隔(10)で抜き出し)
    t_train_data = tf.gather(t, datapoint_list) # タイムステップデータの抜き出し
    x_train_data = tf.gather(x, datapoint_list) # 解析解データの抜き出し

    '''
    # プロット作成
    plt.plot(t, x, color='darkorange', label='Analytic Solution Result')
    plt.scatter(t_train_data, x_train_data, color="lightseagreen", label='Extract Data Point for DDNN trainning')
    plt.title('Analytic Solution Result')
    plt.xlabel("time")
    plt.ylabel("displacement")
    plt.legend()
    plt.show()
    '''
    
    # 出力されたグラフ上で'q'を押すと次のプロットへ
    print("Please press Q-key")

    '''
    STEP 3
    DDNNの構築と学習
    '''
    # DDNNの構成パラメータ
    n_input = 1 
    n_output = 1
    n_neuron = 32
    n_layer = 4

    
    model = Sequential()
    
    model.add(Dense(units=n_neuron, 
                    activation='tanh', 
                    kernel_initializer=GlorotNormal(), 
                    input_shape=(n_input,), 
                    name='H1'))
        
    for i in range(n_layer-1):
        model.add(Dense(units=n_neuron, 
                        activation='tanh', 
                        kernel_initializer=GlorotNormal(), 
                        name='H{}'.format(str(i+2))))
            
    model.add(Dense(units=n_output, name='output'))      
    


    
    # DDNNの学習パラメータ
    epochs = 1000
    lr = 1e-3 # learning_rate
    
    optimizer = optimizers.Adam(learning_rate=lr)
    loss_fc = losses.MeanSquaredError()
    model.compile(loss=loss_fc, optimizer=optimizer, metrics=['accuracy'])
    
    model.summary()
    ers = EarlyStopping(monitor='val_loss', patience=200, verbose=True)

    history = model.fit(x_train_data, t_train_data,
                    epochs=epochs, 
                    batch_size=1, 
                    verbose=True, #学習の進捗を表示
                    validation_split=0.2, 
                    shuffle=True, 
                    callbacks=[ers] #EarlyStoppingをコールバックで設定
                    ) #history.historyにloss,accuracyが格納される
    
    
    
    x_text_data = np.linspace(0, 1, 20) # DDNNの結果テスト用の入力データ作成
    
    t_tetst_data = model.predict(x_text_data)
    
    # プロット作成
    plt.plot(t, x, label='Analytic Solution Result')
    plt.scatter(x_text_data, t_tetst_data, color='green', label='DDNN predict result')
    plt.title('Analytic Solution Result')
    plt.xlabel("time")
    plt.ylabel("displacement")
    plt.legend()
    plt.show()
    
    
    plt.ﬁgure(ﬁgsize=(15, 6))
    plt.subplots_adjust(wspace=0.2)

    plt.subplot(1,2,1)
    plt.plot(history.history['loss'], label='training') 
    plt.plot(history.history['val_loss'], label='validation')
    plt.ylim(0,1)
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    
    plt.subplot(1,2,2)
    plt.plot(history.history['accuracy'], label='training')
    plt.plot(history.history['val_accuracy'], label='validation')
    plt.ylim(0.5,1)
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('accuracy')

    plt.show()
    
    '''
    # DDNNの構成パラメータ
    n_input = 1 
    n_output = 1
    n_neuron = 32
    n_layer = 4
    epochs = 1000
    
    model = nsim.DataDrivenNNs(n_input, n_output, n_neuron, n_layer)
    
    optimizer = optimizers.Adam(learning_rate=1e-3)
    loss_fc = losses.MeanSquaredError()
    model.build(optimizer, loss_fc)
    
    # DDNNs Trainning
    model.train(epochs, t_train_data, x_train_data, 100)
    print("DDNN trainning complete!")

    # Create test points AND predict result
    t_DDNN_input = np.linspace(0, 1, 20) # DDNNの結果テスト用の入力データ作成
    x_pred = model._model.predict(t_DDNN_input) # 予測結果の出力
    '''
    '''
    # プロット作成
    plt.plot(t, x, label='Analytic Solution Result')
    plt.scatter(t_DDNN_input, x_pred, color='green', label='DDNN predict result')
    plt.title('Analytic Solution Result')
    plt.xlabel("time")
    plt.ylabel("displacement")
    plt.legend()
    plt.show()
    '''
    

    
    
if __name__ == "__main__":
    solve_DDNNs()