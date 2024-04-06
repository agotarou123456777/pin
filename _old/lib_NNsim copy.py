import tensorflow as tf
import keras.layers as layers
import keras.initializers as initializers
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation


########################################################################################################
#
# Support Functions
#
########################################################################################################

def MLP(n_input, n_output, n_neuron, n_layer, act_fn='tanh'):
           
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(
            units=n_neuron,
            activation=act_fn,
            kernel_initializer=tf.keras.initializers.GlorotNormal(),
            input_shape=(n_input,),
            name='H1')
    ])
    
    for i in range(n_layer-1):
        model.add(
            tf.keras.layers.Dense(
                units=n_neuron,
                activation=act_fn,
                kernel_initializer=tf.keras.initializers.GlorotNormal(),
                name='H{}'.format(str(i+2))
            ))
        
    model.add(
        tf.keras.layers.Dense(
            units=n_output,
            name='output'
        ))
    return model


class EarlyStopping:
    '''
    早期に学習停止するEarly Stoppingクラスの定義
    '''

    def __init__(self, patience=10, verbose=1):
        '''
        Parameters:
            patience(int) : 監視するエポック数
            verbose(int)  : EarlyStopのコメント出力.出力あり(1),出力なし(0)
        '''

        self.epoch = 0 # 監視中のエポック数のカウンター初期化
        self.pre_loss = float('inf') # 比較対象の損失を無限大'inf'で初期化
        self.patience = patience
        self.verbose = verbose

    def __call__(self, current_loss):
        '''
        Parameters:
            current_loss(float): 1エポック終了後の検証データの損失
        Return:
            True:監視回数の上限までに前エポックの損失を超えた場合
            False:監視回数の上限までに前エポックの損失を超えない場合
        '''

        if self.pre_loss < current_loss:
            self.epoch += 1

            if self.epoch > self.patience:
                if self.verbose:
                    print('early stopping')
                return True
            
        else:
            self.epoch = 0
            self.pre_loss = current_loss
        return False

########################################################################################################
#
# DataDrivenNNs
#
########################################################################################################

class DataDrivenNNs():
    '''
    DataDriven型のNNクラス
    '''

    def __init__(self, n_input, n_output, n_neuron, n_layer, act_fn='tanh'):
        '''
        Input : 
        n_input   : インプット数
        n_output  : アウトプット数
        n_neuron  : 隠れ層のユニット数
        n_layer   : 隠れ層の層数
        act_fn    : 活性化関数
        '''

        self._model = tf.keras.Sequential([
            layers.Dense(units=n_neuron, activation=act_fn, kernel_initializer=initializers.GlorotNormal(), input_shape=(n_input,), name='H1')
            ])
        
        for i in range(n_layer-1):
            self._model.add(layers.Dense(units=n_neuron, activation=act_fn, kernel_initializer=initializers.GlorotNormal(), name='H{}'.format(str(i+2))
                ))
            
        self._model.add(layers.Dense(units=n_output, name='output'))       

    def build(self, optimizer, loss_fn):
        self._optimizer = optimizer
        self._loss_fn = loss_fn
        return self


    def train_step(self, t_input, x_output):
        with tf.GradientTape() as tape:
            x_pred = self._model(t_input)
            loss = self._loss_fn(x_pred,x_output)
        self._gradients = tape.gradient(loss,self._model.trainable_variables)
        self._optimizer.apply_gradients(zip(self._gradients, self._model.trainable_variables))
        self._loss_values.append(loss)
        return self

    def train(self, epochs, t_data, x_data, plot_interval):
        '''
        学習ループ用の関数
        '''
        self._loss_values = []
        self._early_stopping = EarlyStopping(patience=200, verbose=1)
        
        for i in range(epochs):
            
            if i % plot_interval == 0:
                print("epoch : ", i)
                x_pred = self._model(t_data)
            
            self.train_step(t_data, x_data)
            if self._early_stopping(self._loss_values[-1]): #early stoppingの場合ループを抜ける
                break
                   
########################################################################################################
#
# PINNs(Physics-informed neural networks)
#
########################################################################################################

class PhysicsInformedNNs():
    '''
    PINNs(Physics-informed neural networks)型のNNクラス
    '''

    def __init__(self, n_input, n_output, n_neuron, n_layer, epochs, act_fn='tanh'):
        '''
        n_input    : インプット数
        n_output   : アウトプット数
        n_neuron   : 隠れ層のユニット数
        n_layer    : 隠れ層の層数
        act_fn     : 活性化関数
        epochs     : エポック数
        '''
        self.n_input = n_input
        self.n_output = n_output
        self.n_neuron = n_neuron
        self.n_layer = n_layer
        self.epochs = epochs
        self.act_fn = act_fn
        
        
        self.imgs = []
        self.fig = plt.figure(figsize=(6, 4))
        plt.xlabel("time")
        plt.ylabel("displacement")
        self.writerd = animation.PillowWriter(fps=5)
        
    
    def animation(self, t_ana, x_ana, t_input, x_pred, epoch):
        text_epoch = "epoch : " + str(epoch)
        text_epoch = self.fig.text(0.1, 0.9, text_epoch, size = 12, color = "black",fontfamily="Impact", fontweight="bold")
        img_1 = plt.plot(t_ana, x_ana, label='Analytic Solution Result', color='darkorange')
        img_2 = plt.plot(t_input, x_pred, label='DDNN predict result', color="lightseagreen", linestyle='None', marker='o')
        plt.legend(["Analytic Solution Result","DDNN predict"])
        self.imgs.append(img_1 + img_2 + [text_epoch])
        return self
    
    
    def build(self, optimizer, loss_fn, early_stopping):
        self._model = MLP(self.n_input, self.n_output, self.n_neuron, self.n_layer, self.act_fn)
        self._optimizer = optimizer
        self._loss_fn = loss_fn
        self._early_stopping = early_stopping
        return self

    def train_step(self, t_data, x_data, t_pinn, c, k):
        with tf.GradientTape() as tape_total:
            tape_total.watch(self._model.trainable_variables)
            x_pred = self._model(t_data)
            loss1 = self._loss_fn(x_pred, x_data)
            loss1 = tf.cast(loss1, dtype=tf.float32)

            with tf.GradientTape() as tape2:
                tape2.watch(t_pinn)
                with tf.GradientTape() as tape1:
                    tape1.watch(t_pinn)
                    x_pred_pinn = self._model(t_pinn)
                dx_dt = tape1.gradient(x_pred_pinn, t_pinn)
            dx_dt2 = tape2.gradient(dx_dt, t_pinn)

            dx_dt  = tf.cast(dx_dt, dtype=tf.float32)
            dx_dt2 = tf.cast(dx_dt2, dtype=tf.float32)
            x_pred_pinn = tf.cast(x_pred_pinn, dtype=tf.float32)

            loss_physics = dx_dt2 + c * dx_dt + k * x_pred_pinn
            loss2 = 5.0e-4 * self._loss_fn(loss_physics, tf.zeros_like(loss_physics))
            loss2 = tf.cast(loss2, dtype=tf.float32)

            loss = loss1 + loss2

        self._optimizer.minimize(loss, self._model.trainable_variables, tape=tape_total)
        self._loss_values.append(loss)
        return self

    def train(self, t_ana, x_ana, t_data, x_data, t_pinn, c, k, plot_interval):
        self._loss_values = []
        for i in range(self.epochs):
            
            if i % plot_interval == 0:
                print("epoch : ", i)
                x_pred = self._model(t_data)
                self.animation(t_ana, x_ana, t_data, x_pred, i)
                
            self.train_step(t_data, x_data, t_pinn, c, k)
            
            
            if self._early_stopping(self._loss_values[-1]):
                break
        
        ani = animation.ArtistAnimation(self.fig, self.imgs, interval=100)
        ani.save('PINNs_Trainning_Process.gif', writer=self.writerd) 
        plt.show()
