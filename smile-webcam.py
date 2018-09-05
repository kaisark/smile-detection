# -*- coding=utf-8 -*-
# attribution - https://github.com/LiuXiaolong19920720/smile-detection-Python
import cv2

# 人脸检测器 #face detect
facePath = "lbpcascade_frontalface.xml"
faceCascade = cv2.CascadeClassifier(facePath)

# 笑脸检测器
smilePath = "haarcascade_smile.xml"
smileCascade = cv2.CascadeClassifier(smilePath)

camera = cv2.VideoCapture(0)

while True:
	(ret, img) = camera.read()
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	# if we are viewing a video and we did not grab a frame,
	# then we have reached the end of the video
	if not ret:
		break

	# 首先检测人脸，返回的是框住人脸的矩形框
	faces = faceCascade.detectMultiScale(
    	gray,
    	scaleFactor= 1.1,
    	minNeighbors=8,
    	minSize=(55, 55),
    	flags=cv2.CASCADE_SCALE_IMAGE
	)

	# 画出每一个人脸，提取出人脸所在区域
	for (x, y, w, h) in faces:
		cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
		roi_gray = gray[y:y+h, x:x+w]
		roi_color = img[y:y+h, x:x+w]

		# 对人脸进行笑脸检测
		smile = smileCascade.detectMultiScale(
        	roi_gray,
        	scaleFactor= 1.16,
        	minNeighbors=35,
        	minSize=(25, 25),
        	flags=cv2.CASCADE_SCALE_IMAGE
    	)

		# 框出上扬的嘴角并对笑脸打上Smile标签
		for (x2, y2, w2, h2) in smile:
			cv2.rectangle(roi_color, (x2, y2), (x2+w2, y2+h2), (255, 0, 0), 2)
			cv2.putText(img,'Smile',(x,y-7), 3, 1.2, (0, 255, 0), 2, cv2.LINE_AA)

	cv2.imshow('Smile?', img)

	if cv2.waitKey(1) & 0xFF == ord('q'):
	    break

camera.release()
cv2.destroyAllWindows()
