from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import numpy as np
from PIL import Image




# import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
# from scipy.spatial.distance import cdist

# x = np.random.rand(2,1000)
# y = np.random.rand(2,1000)

def vgg16_feature(images,channel=3):
	model = VGG16(weights='imagenet', include_top=False)
	vgg16_feature_list=[]
	if channel==1:
		for i in [0,1]:
			img= np.concatenate((images[i,:,:,None],images[i,:,:,None],images[i,:,:,None]), axis=-1)
			img = Image.fromarray(img).resize((224, 224))
			# img = image.img_to_array(img)
			img = np.expand_dims(img, axis=0)
			img = preprocess_input(img)
			vgg16_feature = model.predict(img)
			vgg16_feature_np = np.array(vgg16_feature)
			vgg16_feature_list.append(vgg16_feature_np.flatten())
	else:
		for i in [0,1]:
			# img = image.load_img(np.squeeze(images[i,:,:,:]), target_size=(224, 224))
			img = Image.fromarray(images[i,:,:,:]).resize((224, 224))
			# img = image.img_to_array(img)
			img = np.expand_dims(img, axis=0)
			img = preprocess_input(img)
			vgg16_feature = model.predict(img)
			vgg16_feature_np = np.array(vgg16_feature)
			vgg16_feature_list.append(vgg16_feature_np.flatten())
	return np.array(vgg16_feature_list)

# print(cosine_similarity(x, y).shape)

# model.summary()

# img_path = 'src/img.png'
# img = image.load_img(img_path, target_size=(224, 224))
# img_data = image.img_to_array(img)
# img_data = np.expand_dims(img_data, axis=0)
# img_data = preprocess_input(img_data)

# vgg16_feature1 = model.predict(img_data)
# vgg16_feature_np = np.array(vgg16_feature1)
# # vgg16_feature_list=vgg16_feature_np.flatten()
# vgg16_feature_list.append(vgg16_feature_np.flatten())
# vgg16_feature_list.append(vgg16_feature_np.flatten())
# vgg16_feature_list_np= np.array(vgg16_feature_list)
# # vgg16_feature2=model.predict(img_data)
# print(cosine_similarity(vgg16_feature_list_np,vgg16_feature_list_np).shape)
# print(vgg16_feature.shape)