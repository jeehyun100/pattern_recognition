import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import PReLU
from tensorflow.keras.layers import ELU
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.initializers import glorot_normal
from tensorflow.keras.initializers import TruncatedNormal

tf.reset_default_graph()
sess = tf.Session()

# 텐서 `c`를 계산합니다.
for _ in range(10):
    #initializer = TruncatedNormal(mean=0.0, stddev=0.5, seed=1)
    initializer = glorot_uniform()
    values = initializer(shape=(2, 2))
    print(sess.run(values))

