from pytesseract import Output
import pytesseract
import cv2
import json


pytesseract.pytesseract.tesseract_cmd = "D:\\installS\\Tesseract-OCR\\tesseract.exe"
custom_config = r'-l san --psm 6'

#load image
image = cv2.imread("original/Sanskrit Text-1.jpg")

#output directory
out_dir ="output/1/"

rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
results = pytesseract.image_to_data(rgb, output_type=Output.DICT, config= custom_config)


res_dict = {}
ROI_number = 1

for i in range(3, len(results["text"])):
	x = results["left"][i]
	y = results["top"][i]
	w = results["width"][i]
	h = results["height"][i]

	conf = int(results["conf"][i])

	if conf <0:
		cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
		ROI = image[y:y+h, x:x+w]
		cv2.imwrite(out_dir+'ROI_{}.png'.format(ROI_number), ROI)
		res_dict["box{}".format(ROI_number)]={
											"top_left": [x, y],
											"top_right": [x+w, y],
											"bottom_left": [x, y+h],
											"bottom_right": [x+w, y+h]
											}
		ROI_number += 1


filename = out_dir+'savedImage.jpg'
cv2.imwrite(filename, image) 
with open(out_dir+'json_data.json', 'w') as fp:
    json.dump(res_dict, fp)

