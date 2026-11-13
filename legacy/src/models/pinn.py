import tensorflow as tf
from tensorflow.keras.layers import Dense

def build_pinn(input_dim, hidden_units, num_layers):

    model = tf.keras.Sequential()
    
    # Input layer
    model.add(tf.keras.layers.InputLayer(input_shape=(input_dim,)))
    
    # Hidden layers
    for _ in range(num_layers):
        model.add(Dense(hidden_units, activation='tanh',
                       kernel_initializer='glorot_normal'))
    
    # Output layer (temperature prediction)
    model.add(Dense(1, activation=None))
    
    return model
