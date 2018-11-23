#-----------------------------------------------------------
# Proceso
#-----------------------------------------------------------
#Import Flask
from flask import Flask, request
from keras.preprocessing import image
from cnn_executor import cargarModelo
import numpy as np

#Initialize the application service
app = Flask(__name__)
global loaded_model, graph
loaded_model, graph = cargarModelo()

#Define a route
@app.route('/')
def main_page():
	return 'Bienvenido a la URP - RNA!'

@app.route('/perrosygatos/', methods=['GET','POST'])
def rayosx():
	return 'Modelo de Perros y Gatos!'

@app.route('/perrosygatos/default/', methods=['GET','POST'])
def default():
	print (request.args)
	# dimensions of our images.
	img_width, img_height = 299, 299
	# Show
	image_name = request.args.get("imagen")
	img_path='../samples/' + image_name
	img = image.load_img(img_path, target_size=(img_width, img_height))
	img = image.img_to_array(img)
	x = np.expand_dims(img, axis=0) * 1./255

	with graph.as_default():
		score = loaded_model.predict(x)
		if score == 1:# >= 0.5
			resultado = 'Es un perro, score: ' + str(score[0][0])
		else:
		    resultado = 'Es un gato, score: ' + str(score[0][0])
		return resultado

# Run de application
app.run(host='0.0.0.0',port=5000)
