
# guardar modelo
model.save('path_to_my_model.h5')  # or model.save('path_to_my_model')



from matplotlib import pyplot as plt #


x_ejerc=[]

x_ejerc.append(img1_r)

img1=cv2.imread('data\\ejercicio_est\\1.jpeg')
img1_r = cv2.resize(img1 ,(100,100))
img1_r=img1_r.astype('float32') ## para poder escalarlo
img1_r /=255
img1_r.shape

img2=cv2.imread('data\\ejercicio_est\\2.jpeg')
img2_r = cv2.resize(img2 ,(100,100))
img2_r=img2_r.astype('float32') ## para poder escalarlo
img2_r /=255
img2_r.shape
x_ejerc.append(img2_r)

x_ejerc_np=np.array(x_ejerc)
x_ejerc_np.shape
modelo.predict(x_ejerc_np)

modelo=joblib.load('/cod/fc_model.pkl')
joblib.dump(modelo, 'fc_model.joblib')

modelo.save('fc_model.h5')


# Loading the model
from tensorflow.keras.models import load_model
model = load_model('path_to_my_model.h5') 