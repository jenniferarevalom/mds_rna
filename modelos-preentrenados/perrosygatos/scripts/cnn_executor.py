#-----------------------------------------------------------
# Cargando modelo de disco
#-----------------------------------------------------------
import tensorflow as tf
from keras.models import model_from_json
import matplotlib.pyplot as plt

def cargarModelo():
    print("Cargando modelo desde el disco ...")
    json_file = open('../model/rx_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("../model/rx_model.h5")
    loaded_model.compile(optimizer='adam',decay=0.0), loss='binary_crossentropy', metrics=['accuracy'])
    print("Modelo cargado de disco!")
    graph = tf.get_default_graph()
    return loaded_model, graph
