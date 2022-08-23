from tensorflow.keras.regularizers import l2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend


D = 2
L = 12
delta_t = 25
K = 64
M = 4
d = 3
batch_size = 32

wd = l2(0.0)


def neg_padding(tensor):
    return tf.pad(
        tensor,
        tf.constant(
            [
                [0, 0],
                [1, 1],
                [0, 0]
            ]
        ),
        mode='CONSTANT', constant_values=-1
    )


new_layer = layers.Lambda(neg_padding)


def get_subnetwork():
    # delta_t can change
    input_layer = layers.Input(shape=(None, D))  # b_s, delta_t, D

    # Как сделать паддинг -1???
    # Как добавить Гауссовский дропаут
    x = new_layer(input_layer)  # b_s, delta_t + 2, L
    x = layers.Conv1D(L, 3, padding='valid')(x)  # b_s, delta_t, L
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Bidirectional(
        layers.LSTM(units=K, return_sequences=True)
    )(x)  # b_s, delta_t, 2 * K

    x = layers.Bidirectional(
        layers.LSTM(units=K, return_sequences=True)
    )(x)  # b_s, delta_t, 2 * K

    x = layers.Bidirectional(
        layers.LSTM(units=K)
    )(x)  # b_s, 2 * K

    x = layers.BatchNormalization()(x)
    x = layers.Dense(K)(x)  # b_s, K
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dense(1, activation='relu')(x)  # b_s, 1

    final = x

    subnetwork = keras.Model(inputs=input_layer, outputs=final)
    return subnetwork


def special_reshape_1(x):
    #     delta_t can change
    shape = [tf.shape(x)[k] for k in range(4)]
    x_reshaped = tf.reshape(x, [shape[0] * shape[1], shape[2], shape[3]])
    return x_reshaped


def special_reshape_2(x):
    return backend.reshape(x, (-1, (M + 1) ** d, 1))


special_reshape_layer_in = layers.Lambda(special_reshape_1)
special_reshape_layer_out = layers.Lambda(special_reshape_2)


def get_full_model():
    tf.keras.backend.clear_session()
    subnetwork = get_subnetwork()
    subnetwork.trainable = True
    input_layer_n = layers.Input(shape=((M + 1) ** d, None, D))  # b_s, (M+1)**d, delta_t, D
    flattened_tensor = special_reshape_layer_in(input_layer_n)  # b_s * (M+1)**d, delta_t, D

    predicted_from_sn = subnetwork(flattened_tensor)  # b_s * (M+1)**d, 1

    unflattend_prediction = special_reshape_layer_out(predicted_from_sn)  # b_s, (M+1)**d, 1

    unflattend_prediction.shape

    x = layers.MaxPooling1D(pool_size=(M + 1) ** (d - 1))(unflattend_prediction)  # b_s, M + 1, 1
    # print(x.shape)
    x = layers.Flatten()(x)  # b_s, M + 1

    dense1_1 = layers.Dense(K)(x)
    dense1_1 = layers.BatchNormalization()(dense1_1)
    dense1_1 = layers.ReLU()(dense1_1)
    dense1_2 = layers.Dense(K)(dense1_1)
    dense1_2 = layers.BatchNormalization()(dense1_2)
    dense1_2 = layers.ReLU()(dense1_2)
    dense1_3 = layers.Dense(M + 1, activation='softmax')(dense1_2)

    dense2_1 = layers.Dense(K)(x)
    dense2_1 = layers.BatchNormalization()(dense2_1)
    dense2_1 = layers.ReLU()(dense2_1)
    dense2_2 = layers.Dense(K, activation='relu')(dense2_1)
    dense2_2 = layers.BatchNormalization()(dense2_2)
    dense2_2 = layers.ReLU()(dense2_2)
    dense2_3 = layers.Dense(2, activation='softmax')(dense2_2)

    full_model = keras.Model(
        inputs=input_layer_n, outputs=[dense1_3, dense2_3]
    )
    return full_model