import tensorflow as tf
import keras.layers as layers
import keras.initializers as initializers
import numpy as np
import tensorflow as tf
from keras.losses import MeanSquaredError
from keras.optimizers import Adam
from keras.layers import Dense
from keras.models import Sequential
from keras.initializers import GlorotNormal
import matplotlib.pyplot as plt
import matplotlib.animation as animation



                   
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
        self.n_input = n_input
        self.n_output = n_output
        self.n_neuron = n_neuron
        self.n_layer = n_layer
        self.epochs = epochs
        self.act_fn = act_fn
    
    def build(self, optimizer, loss_fn, early_stopping):
        n_input = 1 
        n_output = 1
        n_neuron = 32
        n_layer = 4
        plot_interval = 100
        
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
