# USAGE
# python recognize_gambar.py --detector face_detection_model \
#	--embedding-model openface_nn4.small2.v1.t7 \
#	--recognizer output/recognizer.pickle \
#	--le output/le.pickle --image images/afifa1.jpg

# impor paket yang diperlukan
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os

# membangun parser argumen dan parsing argumen
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-d", "--detector", required=True,
	help="path to OpenCV's deep learning face detector")
ap.add_argument("-m", "--embedding-model", required=True,
	help="path to OpenCV's deep learning face embedding model")
ap.add_argument("-r", "--recognizer", required=True,
	help="path to model trained to recognize faces")
ap.add_argument("-l", "--le", required=True,
	help="path to label encoder")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# muat detektor wajah bersambung kami dari disk
print("[INFO] memuat detektor wajah...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# memuat model penyisipan wajah berseri dari serial
print("[INFO] memuat pengenal wajah...")
embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])

# muat model pengenalan wajah yang sebenarnya bersama dengan label enkoder
recognizer = pickle.loads(open(args["recognizer"], "rb").read())
le = pickle.loads(open(args["le"], "rb").read())

# muat gambar, ubah ukurannya menjadi lebar 600 piksel (sementara
# mempertahankan rasio aspek), lalu ambil dimensi gambar
image = cv2.imread(args["image"])
image = imutils.resize(image, width=400)
(h, w) = image.shape[:2]

# membangun gumpalan dari gambar
imageBlob = cv2.dnn.blobFromImage(
	cv2.resize(image, (300, 300)), 1.0, (300, 300),
	(104.0, 177.0, 123.0), swapRB=False, crop=False)

# menerapkan pendeteksi wajah berbasis pembelajaran OpenCV yang mendalam untuk melokalisasi
# wajah pada gambar input
detector.setInput(imageBlob)
detections = detector.forward()

# loop atas deteksi
for i in range(0, detections.shape[2]):
	# ekstrak kepercayaan (mis., probabilitas) yang terkait dengan
    # prediksi
	confidence = detections[0, 0, i, 2]

	# saring deteksi lemah
	if confidence > args["confidence"]:
		# menghitung (x, y) - koordinat kotak pembatas untuk
        # wajah
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")

		# ekstrak ROI wajah
		face = image[startY:endY, startX:endX]
		(fH, fW) = face.shape[:2]

		# Pastikan lebar dan tinggi wajah cukup besar
		if fW < 20 or fH < 20:
			continue

		# buat gumpalan untuk ROI wajah, lalu lewati gumpalan
        # melalui model penyisipan wajah kami untuk mendapatkan 128-d
        # kuantifikasi wajah
		faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96),
			(0, 0, 0), swapRB=True, crop=False)
		embedder.setInput(faceBlob)
		vec = embedder.forward()

		# melakukan klasifikasi untuk mengenali wajah
		preds = recognizer.predict_proba(vec)[0]
		j = np.argmax(preds)
		proba = preds[j]
		name = le.classes_[j]

		# menggambar kotak pembatas wajah bersama dengan yang terkait
        # probabilitas
		text = "{}: {:.2f}%".format(name, proba * 100)
		y = startY - 10 if startY - 10 > 10 else startY + 10
		cv2.rectangle(image, (startX, startY), (endX, endY),
			(0, 255, 0), 2)
		cv2.putText(image, text, (startX, y),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 2)

# tampilkan gambar output
cv2.imshow("Hasil", image)
cv2.waitKey(0)