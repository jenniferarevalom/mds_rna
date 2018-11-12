# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 18:53:49 2018

@author: JENNIFER
"""

# Credits:
# https://link.springer.com/article/10.1007/s10278-018-0079-6
# https://github.com/ImagingInformatics/machine-learning
# https://github.com/paras42/Hello_World_Deep_Learning

# -----------------------------------------------------------
# Cargando modelo de disco
import tensorflow as tf
from keras.models import model_from_json
import matplotlib.pyplot as plt


def cargarModelo():
    json_file = open('../modelo/jenn_cnn.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("../modelo/jenn_cnn.h5")
    print("Cargando modelo desde el disco ...")
    loaded_model.compile(optimizer="Adam", loss='binary_crossentropy', metrics=['accuracy'])
    print("Modelo cargado de disco!")
    graph = tf.get_default_graph()
    return loaded_model, graph