# Credits:
# https://link.springer.com/article/10.1007/s10278-018-0079-6
# https://github.com/ImagingInformatics/machine-learning
# https://github.com/paras42/Hello_World_Deep_Learning

# -----------------------------------------------------------
# Cargando modelo de disco
from keras.models import model_from_json
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from keras.optimizers import Adam

# dimensions of our images.
img_width, img_height = 299, 299

json_file = open('./model/rx_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("./model/rx_model.h5")
print("Cargando modelo desde el disco ...")
loaded_model.compile(optimizer=Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08,decay=0.0), loss='binary_crossentropy', metrics=['accuracy'])
print("Modelo cargado de disco!")

# Show
img_path='dataset/samples/img1.png' #change to location of chest x-ray
img = image.load_img(img_path, target_size=(img_width, img_height))

# Predict
plt.imshow(img)
plt.show()
img = image.img_to_array(img)
x = np.expand_dims(img, axis=0) * 1./255
score = loaded_model.predict(x)
print('Prediccion:', score, 'Abdomen X-ray' if score < 0.5 else 'Pulmon X-ray')