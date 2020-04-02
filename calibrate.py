import cv2
import os
import numpy as np
def GetImages(pathIn):

	frame_array = []
	files = [f for f in os.listdir(
		pathIn) if os.path.isfile(os.path.join(pathIn, f))]

	# #for sorting the file names properly
	# files.sort(key=lambda x: int(x[:-4]))
	print(files)

	for i in range(len(files)):
		filename = os.path.join(pathIn, files[i])
		#reading each files
		img = cv2.imread(filename)
		#inserting the frames into an image array
		frame_array.append(img)

	return frame_array


def PlotCorners(img, corners):
	main_corners = []
	for i, corner in enumerate(corners):
		if(i==0 or i==8 or i==45 or i==53 ):
			x = corner[0][0]
			y = corner[0][1]
			cv2.circle(img, (x, y), 3, (0, 0, 255), 4)
			main_corners.append([x, y])
	return img,np.array(main_corners)


def findH(model_xy, main_corners):
	# cv2.findChessboardCorners finds corners indices interating thoruhg x first than then y axis
	H, status = cv2.findHomography(model_xy, main_corners)
	return H
	

images = GetImages('./input')

corners = []
chessboard_flags = cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE

for i,img in enumerate(images):


	ret, corners = cv2.findChessboardCorners(img, (9, 6), chessboard_flags)

	img,main_corners = PlotCorners(img, corners)
	model_xy = np.array([[21.5, 21.5],
							[21.5*9,21.5],
							[21.5,21.5*6],
							[21.5*9, 21.5*6]], dtype='float32')
	H = findH(model_xy, main_corners)
	# H_inv = np.linalg.inv(H)
	warped_img = cv2.warpPerspective(
		img, H_inv, dsize=(img.shape[0], img.shape[1]))
	cv2.imwrite('warped_output/img'+str(i)+'.jpg', warped_img)

	cv2.imwrite('output/img'+str(i)+'.jpg', img)
