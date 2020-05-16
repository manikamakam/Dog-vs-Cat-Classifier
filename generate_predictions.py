# import cv2
import glob
# import matplotlib.pyplot as plt
import re
import csv
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

def rename():
	img_paths = glob.glob("data/test1/*")
	for name in sorted(img_paths):
		print(name)
		res = re.findall("(\d+).jpg", name)
		if not res: 
			continue
		# print res[0] # You can append the result to a list
		img = cv2.imread(name)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		plt.imsave('data/test/%05d.png' % int(res[0]),img)


 

with open('predictions_test.csv', 'a') as file:
	writer = csv.writer(file,delimiter=',')
	writer.writerow(('id', 'label'))
def predictions():
	img_paths = glob.glob("data/test/*")
	for name in sorted(img_paths):
		# print(name)
		img = load_img(name, target_size=(200, 200))
		img = img_to_array(img)
		img = img.reshape(1, 200, 200, 3)
		img = img.astype('float32')

		res = re.findall("(\d+).png", name)
		if not res: 
			print("dont go hereeeeeeeeeeeeee")
			continue
		# print res[0]

		model = load_model('model2.h5')
		result = model.predict(img)

		temp = [int(res[0]), int(result[0][0])]
		print(temp)
		with open('predictions_test.csv', 'a') as file:
			writer = csv.writer(file,delimiter=',')

			writer.writerow((int(res[0]), int(result[0][0])))

predictions()