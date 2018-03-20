'''

DSP_LAB Final Project done by Yun Gao & Jingjie Sheng.
We made one selfie-aid APP which can both take photos and record short videos.
We provided many fun effects for users to play with.
All other needed directions will be shown when you run the program.

'''

import sys
if sys.version_info[0] < 3:
	import Tkinter as Tk 	# for Python 2
else:
	import tkinter as Tk   	# for Python 3
import numpy as np
import cv2
from PIL import Image, ImageTk
import datetime
import os
from time import sleep
from scipy.spatial import distance

class DSP_PROJ:
	def __init__(self):

		global stopFlag, panel, stopFlag_bw, direction, COLOR1, COLOR2, ix, iy, drawing, Beautyflags
		# stopFlag = True
		# stopFlag_bw = True
		Beautyflags = False
		COLOR1 = (218, 246, 174)
		COLOR2 = (192, 180, 92)
		ix = -1
		iy = -1
		drawing = False

		self.top = Tk.Tk()  
		self.top.title('Selfie-aid APP')

		# Frame_1
		self.F1 = Tk.Frame(self.top,bg = "light goldenrod")
		self.F1.pack()
		# Labels about information
		self.L1 = Tk.Label(self.F1, text = 'WELCOME TO OUR APP',fg = 'orange', bg = 'lemon chiffon', font=("Courier", 30, "bold"))
		self.L1.pack()
		self.L2 = Tk.Label(self.F1, text = '-----* a magic selfie-aid app that provides you with fun effects *-----',fg = 'seashell4', bg = 'lemon chiffon', font = ("Courier", 12, "italic"))
		self.L2.pack()
		lace = ImageTk.PhotoImage(Image.open("lace1.gif"))
		self.L2_2 = Tk.Label(self.F1, image = lace)
		self.L2_2.pack()
		# Directions
		self.L2_3 = Tk.Label(self.F1, font = ("Courier", 14, "bold"), text = "-----#####-----<   Directions  >-----#####-----",fg = 'seashell4',bg = "lemon chiffon")
		self.L2_3.pack()
		self.L2_4 = Tk.Label(self.F1, font = ("Courier", 14), text = "Please choose mode first.\n < Snapshot >  OR  < Short Video >\n", fg = 'seashell4', bg = "lemon chiffon")
		self.L2_4.pack()

		# Frame_2
		self.F2 = Tk.Frame(self.top,bg = "lemon chiffon")
		self.F2.pack()
		# Buttons
		self.B1 = Tk.Button(self.F2, text = 'Snapshot!', font = ('Courier New bold',), command = self.Snapshot)
		self.B2 = Tk.Button(self.F2, text = 'Video!', font = ('Courier New bold',), command = self.Video_Direction)
		self.B3 = Tk.Button(self.F2, text = 'Quit!', font = ('Courier New bold',), command = self.top_STOP)
		self.B1.pack(side = 'left')
		self.B2.pack(side = 'left')
		self.B3.pack(side = 'right')

		self.F3 = Tk.Frame(self.top,bg = "lemon chiffon")
		self.F3.pack()
		self.L2 = Tk.Label(self.F3, text = '    ',fg = 'seashell4', bg = 'white', font = ("Courier", 12, "italic"))
		self.L2.pack()
		
		self.top.mainloop()

	def top_STOP(self):

 		self.top.quit()

	def Snapshot(self):

		self.ss_top = Tk.Toplevel()  
		self.ss_top.title('Snapshot!')
		self.cap = cv2.VideoCapture(0)

		# Frame_1
		self.ss_F1 = Tk.Frame(self.ss_top,bg = "royal blue")
		self.ss_F1.pack()
		# Labels about information
		self.L1 = Tk.Label(self.ss_F1, text = 'Snapshot Direction',fg = 'midnight blue', bg = 'alice blue', font=("Courier", 30, "bold"))
		self.L1.pack()
		self.L2 = Tk.Label(self.ss_F1, text = '-----* a magic selfie-aid app that provides you with fun effects *-----',fg = 'seashell4', bg = 'LightSkyBlue1', font = ("Courier", 12, "italic"))
		self.L2.pack()

		self.ss_F3 = Tk.Frame(self.ss_top,bg = 'alice blue')
		self.ss_F3.pack()
		# save place for video
		self.ss_panel = Tk.Label(self.ss_F3, text = "Let's begin.", font=("Courier", 24, "bold"),fg = 'RoyalBlue4',bg = 'alice blue')  # initialize image panel
		self.ss_panel.pack(padx=10, pady=10,side = 'right')

		# Frame_4
		self.ss_F4 = Tk.Frame(self.ss_F3, bg = 'alice blue')
		self.ss_F4.pack(side = 'right')

		k1 = Tk.DoubleVar()
		k1.set(1.0)
		k2 = Tk.StringVar()
		k2.set(str(k1.get()))

		# Directions
		direction1 = Tk.Label(self.ss_F4, font = ("Courier", 14, "bold"), text = "\n-----#####-----<   CLICK ON START  >-----#####-----\n-----#####-----<   THEN CLICK ON SNAPSHOT  >-----#####-----\n",fg = 'seashell4',bg = 'alice blue')
		direction2 = Tk.Label(self.ss_F4, text = "R----blue rectangle with mouse moving.",fg = 'SlateBlue4', anchor="e",bg = 'alice blue')
		direction19 = Tk.Label(self.ss_F4, text = "E----blue circle frame with mouse moving.",fg = 'SlateBlue3', anchor="e",bg = 'alice blue')
		direction3 = Tk.Label(self.ss_F4, text = "F----recognize your face&eyes.",fg = 'SteelBlue4', anchor="e",bg = 'alice blue')
		direction4 = Tk.Label(self.ss_F4, text = "C----any color of circle with mouse click.",fg = 'SlateBlue2', anchor="e",bg = 'alice blue')
		direction5 = Tk.Label(self.ss_F4, text = "Q----SAVE & QUIT.\n\n", fg = 'DarkOliveGreen3', anchor="w",bg = 'alice blue') 
		direction6 = Tk.Label(self.ss_F4, text = "S----sunglasses on your face.", fg = 'SteelBlue3', anchor="w",bg = 'alice blue') 
		direction7 = Tk.Label(self.ss_F4, text = "A----hat sticker with mouse click.",fg = 'RoyalBlue2', anchor="w",bg = 'alice blue') 
		direction8 = Tk.Label(self.ss_F4, text = "L----draw color lines with mouse moving.", fg = 'RoyalBlue4', anchor="w",bg = 'alice blue') 
		direction9 = Tk.Label(self.ss_F4, text = "M----draw rainbow with mouse moving.",fg = 'RoyalBlue3', anchor="w",bg = 'alice blue') 
		direction10 = Tk.Label(self.ss_F4, text = "T----return to original photo.",fg = 'DarkOliveGreen4', anchor="w",bg = 'alice blue') 
		direction11 = Tk.Label(self.ss_F4, text = "###-----RETURN-----###", bg = 'light blue')
		direction12 = Tk.Label(self.ss_F4, text = "###-----DRAWING-----###", bg = 'light blue')
		direction13 = Tk.Label(self.ss_F4, text = "###-----FACE REG-----###", bg = 'light blue')
		direction14 = Tk.Label(self.ss_F4, text = "###-----STICKERS-----###", bg = 'light blue')
		direction15 = Tk.Label(self.ss_F4, text = "###-----SAVE IMG-----###", bg = 'light blue')
		direction16 = Tk.Label(self.ss_F4, text = "Y----turn to Mickey Mouse.", fg = 'SkyBlue4', anchor="w",bg = 'alice blue')
		direction17 = Tk.Label(self.ss_F4, text = "###-----FIlTER-----###", bg = 'light blue')
		direction18 = Tk.Label(self.ss_F4, text = "B----blur the photo.", fg = 'DarkOrchid4', anchor="w",bg = 'alice blue')
		direction20 = Tk.Label(self.ss_F4, text = "N----brighten the photo.", fg = 'DarkOrchid3', anchor="w",bg = 'alice blue')		
		direction21 = Tk.Label(self.ss_F4, text = "D----cartoon glasses on your face.", fg = 'SteelBlue2', anchor="w",bg = 'alice blue') 	
		direction22 = Tk.Label(self.ss_F4, text = "U----turn to Nick.", fg = 'SkyBLue3', anchor="w",bg = 'alice blue')
		direction23 = Tk.Label(self.ss_F4, text = "I----turn to Tigger.", fg = 'DarkSlateGray4', anchor="w",bg = 'alice blue')
		direction24 = Tk.Label(self.ss_F4, text = "###-----CARTOON FIG-----###", bg = 'light blue')
		
		# instructions
		direction1.pack()
		# filters
		direction17.pack()
		direction18.pack()
		direction20.pack()
		# drawing	
		direction12.pack()
		direction2.pack()
		direction19.pack()
		direction4.pack()
		# stickers
		direction14.pack()
		direction8.pack()
		direction9.pack()
		direction7.pack()
		# face recognitions
		direction13.pack()
		direction3.pack()
		direction6.pack()
		direction21.pack()
		# cartoon figures
		direction24.pack()
		direction16.pack()
		direction22.pack()
		direction23.pack()
		# return
		direction11.pack()
		direction10.pack()
		# save & quit
		direction15.pack()
		direction5.pack()

		# Buttons
		self.ss_F2 = Tk.Frame(self.ss_top)
		self.ss_F2.pack()
		self.B1 = Tk.Button(self.ss_F2, text = 'Start!', font = ('Courier New bold',), command = self.START)
		self.B2 = Tk.Button(self.ss_F2, text = 'Quit!', font = ('Courier New bold',), command = self.ss_STOP)
		self.B3 = Tk.Button(self.ss_F2, text = 'Snapshot!', font = ('Courier New bold',), command = self.take_snapshot)
		self.B1.pack(side = 'left')
		self.B2.pack(side = 'right')
		self.B3.pack(side = 'right')

		self.ss_F5 = Tk.Frame(self.ss_top)
		self.ss_F5.pack(side = 'bottom')
		self.L3 = Tk.Label(self.ss_F5, text = '    ',fg = 'seashell4', bg = 'white', font = ("Courier", 12, "italic"))
		self.L3.pack()

	def START(self):
		# After choosing 'Start!' on Snapshot Mode
		self.STARTLOOP()
		sleep(0.1)
		self.top.after(100,self.START)

	def STARTLOOP(self):
		# Run camera & Display on the window
		global Beautyflags
		ok, frame = self.cap.read()
		if ok:  
			height, width = frame.shape[:2]
			frame = cv2.resize(frame,(int(2*width/3), int(2*height/3)), interpolation = cv2.INTER_CUBIC)
			cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)  # convert colors from BGR to RGBA 
			self.current_image = Image.fromarray(cv2image)  # convert image for PIL
			imgtk = ImageTk.PhotoImage(image=self.current_image)  # convert image for tkinter
			self.ss_panel.imgtk = imgtk  # anchor imgtk so it does not be deleted by garbage-collector
			self.ss_panel.config(image=imgtk)  # show the image
			# print('ok!!!loop')
			Beautyflags = True

	def ss_STOP(self):
		# Stop Snapshot
		self.cap.release()
		cv2.destroyAllWindows()
		self.top.quit()

	def rv_STOP(self):
		# Stop video recording
		self.rv_top.quit()
		self.top.quit()

	def take_snapshot(self):
		# Save photo & call IMG_PROCESSING() to process photo
		output_path = "./"
		ts = datetime.datetime.now() # grab the current timestamp
		self.filename = "{}.png".format(ts.strftime("%Y-%m-%d_%H-%M-%S"))  # construct filename
		p = os.path.join(output_path, self.filename)  # construct output path
		self.current_image.save(p, "PNG")  # save image as jpeg file
		print("[INFO] saved {}".format(self.filename))
		self.IMG_PROCESSING()

	# Functions be used to process photo:
	
	def beautybrightness(self,img, gamma):
		# Add 'Beauty' Filter to photo
		invGamma = 1.0 / gamma
		table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
		return cv2.LUT(img, table)
	
	def draw_rectangle(self,event,x,y,params,flag):
		# Draw a rectangle with mouse clicking and moving
		global ix, iy, drawing
		if event == cv2.EVENT_LBUTTONDOWN: 
			ix = x
			iy = y
			drawing = True
		elif event == cv2.EVENT_MOUSEMOVE: 
			if drawing == True:
				cv2.rectangle(img, (ix,iy), (x,y), COLOR1, -1)
		elif event == cv2.EVENT_LBUTTONUP: 
			drawing = False
			cv2.rectangle(img, (ix,iy), (x,y), COLOR1,-1)

	def draw_circle_frame(self,event,x,y,params,flag):
		# Draw a circle frame with mouse clicking and moving
		global ix, iy, drawing
		if event == cv2.EVENT_LBUTTONDOWN:   
			ix = x
			iy = y
			drawing = True
			cv2.circle(img, (ix,iy), int(distance.euclidean((ix,iy),(x,y))), COLOR2,2)
		elif event == cv2.EVENT_LBUTTONUP:        
			dst = int(distance.euclidean((ix,iy),(x,y)))
			cv2.circle(img, (ix,iy), dst, COLOR2,2)

	def nothing(self,x):
		# Used when adding Trackbars in IMG_PROCESSING()
		pass

	def draw_circle(self,event,x,y,flag,params):
		# Draw a filled circle with any color you want
		global switch0,switch1,switch2
		b = cv2.getTrackbarPos(switch0, 'Create your own fun selfie!')
		g = cv2.getTrackbarPos(switch1, 'Create your own fun selfie!')
		r = cv2.getTrackbarPos(switch2, 'Create your own fun selfie!')
		
		RADIUS = 20    			
		if event == cv2.EVENT_LBUTTONDOWN: 
			cv2.circle(img,(x,y), RADIUS, [b,g,r], -1)      

	def draw_line(self,event,x,y,params,flag):
# Draw a colorful line with color changing with position
		global ix, iy, drawing
		if event == cv2.EVENT_LBUTTONDOWN:    
			ix = x
			iy = y
			drawing = True
			img[y-3:y+3,x-3:x+3] = [255-x,255-y,255-9*y]
		elif event == cv2.EVENT_MOUSEMOVE:      
			if drawing == True:
				img[y-3:y+3,x-3:x+3] = [255-x,255-y,255-9*y]
		elif event == cv2.EVENT_LBUTTONUP:     
			drawing = False
			img[y-3:y+3,x-3:x+3] = [255-x,255-y,255-9*y]

	def sticker_click(self,event,x,y,flag,params):
		s_img = cv2.imread('hat.png',-1)
		mask = s_img[:,:,3]
		mask_inv = cv2.bitwise_not(mask)
		s_img = s_img[:,:,0:3]
		if event == cv2.EVENT_LBUTTONDOWN:
			roi_color = img[0:img.shape[0], 0:img.shape[1]]
			roi = roi_color[y:y+s_img.shape[0], x:x+s_img.shape[1]]
			print(roi.shape)
			print(mask_inv.shape)
			roi_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
			roi_fg = cv2.bitwise_and(s_img,s_img,mask = mask)
			dst = cv2.add(roi_bg,roi_fg)
			roi_color[y:y+s_img.shape[0], x:x+s_img.shape[1]] = dst
	def sticker_line(self,event,x,y,params,flag):
# Adding stickers with your mouse moving
		global ix, iy, drawing, fsflags, fcflags, scflags
		s_img = cv2.imread("rainbow.png",1)
		if event == cv2.EVENT_LBUTTONDOWN:    
			ix = x
			iy = y
			drawing = True
		elif event == cv2.EVENT_MOUSEMOVE:     
			if drawing == True:
				img[y:y+s_img.shape[0], x:x+s_img.shape[1]] = s_img
		elif event == cv2.EVENT_LBUTTONUP:       
			drawing = False

	def smooth_img(self, img):	
		# Blur the photo
		kernel = np.ones((3,3), np.float32)/9
		img = cv2.filter2D(img,-1,kernel)
		return img

	def brightness(self,img, gamma):
		# Brighten the photo
		invGamma = 1.0 / gamma
		table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
		return cv2.LUT(img, table)

	def IMG_PROCESSING(self):
		# Show photo in a new CV2 window where photo can be processed
		cv2.namedWindow('Create your own fun selfie!')
		global img, roi_color,roi_gray, scflags, fcflags
		img = cv2.imread(self.filename)
		if Beautyflags == True:
			img = self.beautybrightness(img, 1.4)

		# # Flag for adding Cartoonglasses
		# scflags = False
		# # Flag for adding sunglasses and Cartoon characters
		# fcflags = False
		height = img.shape[0]
		width = img.shape[1]
		
		# Preprocess for face recognization
		self.faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
		self.eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		faces = self.faceCascade.detectMultiScale(gray,scaleFactor=1.15,minNeighbors=5,minSize=(30, 30),flags = cv2.CASCADE_SCALE_IMAGE)
		eyes = 0
		for (x, y, w, h) in faces:
			roi_gray = gray[y:y+h, x:x+w]
			roi_color = img[y:y+h, x:x+w]
			eyes = self.eye_cascade.detectMultiScale(roi_gray)
		
		# Display the image 
		cv2.imshow('Create your own fun selfie!', img)	
			
		while True:
			cv2.imshow('Create your own fun selfie!', img)

			key = cv2.waitKey(1)                      
			
			if key == ord('q'):
				break
			# Draw rectangle
			if key == ord('r'):	
				cv2.setMouseCallback('Create your own fun selfie!', self.draw_rectangle)       
			# Draw circle frame
			if key == ord('e'):	
				cv2.setMouseCallback('Create your own fun selfie!', self.draw_circle_frame)       
			# Draw circle with any colors
			if key == ord('c'):	
				global switch0,switch1,switch2
				switch0 = 'blue'
				switch1 = 'green'
				switch2 = 'red'
				cv2.createTrackbar(switch0,'Create your own fun selfie!',255,255,self.nothing)
				cv2.createTrackbar(switch1,'Create your own fun selfie!',255,255,self.nothing)
				cv2.createTrackbar(switch2,'Create your own fun selfie!',255,255,self.nothing)
				
				cv2.setMouseCallback('Create your own fun selfie!', self.draw_circle)       
			# Draw colorful lines
			if key == ord('l'):	
				cv2.setMouseCallback('Create your own fun selfie!', self.draw_line)       
			# Face recognization
			if key == ord('f'):	
				for (x, y, w, h) in faces:
					cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
				if type(eyes) == int:
					pass
				else:
					for (ex, ey, ew, eh) in eyes:
						cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(200,255,100),2)
			
# Sunglasses on face - fcflags
			if key == ord('s'):
				s_img = cv2.imread('glasses.png',-1)
				orig_mask = s_img[:,:,3]
				orig_mask_inv = cv2.bitwise_not(orig_mask)
				s_img = s_img[:,:,0:3]
				origHeight, origWidth = s_img.shape[:2]
				gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
				for (x, y, w, h) in faces:
					roi_color = img[0:height, 0:width]
					s_img_Width = int(w)
					s_img_Height = int(1.0*w/origWidth*origHeight)
					s_img = cv2.resize(s_img, (s_img_Width,s_img_Height),interpolation = cv2.INTER_AREA)
					mask = cv2.resize(orig_mask, (s_img_Width,s_img_Height), interpolation = cv2.INTER_AREA)
					mask_inv = cv2.resize(orig_mask_inv, (s_img_Width,s_img_Height), interpolation = cv2.INTER_AREA)					
					roi = roi_color[y+int(w/4):y+int(w/4)+s_img.shape[0], x:x+s_img.shape[1]]
					roi_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
					roi_fg = cv2.bitwise_and(s_img,s_img,mask = mask)
					dst = cv2.add(roi_bg,roi_fg)
					roi_color[y+int(w/4):y+int(w/4)+s_img.shape[0], x:x+s_img.shape[1]] = dst
			
# Cartoonglasses on face - fcflags
			if key == ord('d'):
				s_img = cv2.imread('glasses1.png',-1)
				orig_mask = s_img[:,:,3]
				orig_mask_inv = cv2.bitwise_not(orig_mask)
				s_img = s_img[:,:,0:3]
				origHeight, origWidth = s_img.shape[:2]
				gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
				for (x, y, w, h) in faces:
					roi_color = img[0:height, 0:width]
					s_img_Width = int(w)
					s_img_Height = int(1.0*w/origWidth*origHeight)
					s_img = cv2.resize(s_img, (s_img_Width,s_img_Height),interpolation = cv2.INTER_AREA)
					mask = cv2.resize(orig_mask, (s_img_Width,s_img_Height), interpolation = cv2.INTER_AREA)
					mask_inv = cv2.resize(orig_mask_inv, (s_img_Width,s_img_Height), interpolation = cv2.INTER_AREA)					
					roi = roi_color[y+int(w/4):y+int(w/4)+s_img.shape[0], x:x+s_img.shape[1]]
					roi_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
					roi_fg = cv2.bitwise_and(s_img,s_img,mask = mask)
					dst = cv2.add(roi_bg,roi_fg)
					roi_color[y+int(w/4):y+int(w/4)+s_img.shape[0], x:x+s_img.shape[1]] = dst
			
# Mickey Mouse on face - fcflags
			if key == ord('y'):
				s_img = cv2.imread('mickey.png',-1)
				orig_mask = s_img[:,:,3]
				orig_mask_inv = cv2.bitwise_not(orig_mask)
				s_img = s_img[:,:,0:3]
				origHeight, origWidth = s_img.shape[:2]
				gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
				for (x, y, w, h) in faces:
					height = img.shape[0]
					width = img.shape[1]
					roi_color = img[0:height, 0:width]
					s_img_Width = int(1.7*w)
					s_img_Height = int(1.7*h)
					y_ = float(s_img_Height-h)*0.96
					x_ = float(s_img_Width-w)/2
					y_1 = y-int(y_)
					x_1 = x-int(x_)
					y_2 = y_1+s_img_Height
					x_2 = x_1+s_img_Width
						
					if y_1 < 0:
						y_1 = 0
					if x_1 < 0:
						x_1 = 0
					if y_2 > height:
						y_2 = height
					if x_2 > width:
						x_2 = width

					s_img_Width = int(x_2-x_1)
					s_img_Height = int(y_2-y_1)
					s_img = cv2.resize(s_img, (s_img_Width,s_img_Height),interpolation = cv2.INTER_AREA)
					mask = cv2.resize(orig_mask, (s_img_Width,s_img_Height), interpolation = cv2.INTER_AREA)
					mask_inv = cv2.resize(orig_mask_inv, (s_img_Width,s_img_Height), interpolation = cv2.INTER_AREA)
					
					roi = roi_color[y_1:y_2, x_1:x_2]
					roi_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
					roi_fg = cv2.bitwise_and(s_img,s_img,mask = mask)
					dst = cv2.add(roi_bg,roi_fg)
					roi_color[y_1:y_2, x_1:x_2] = dst
# Nick on face - fcflags	
			if key == ord('u'):
				s_img = cv2.imread('nick.png',-1)
				orig_mask = s_img[:,:,3]
				orig_mask_inv = cv2.bitwise_not(orig_mask)
				s_img = s_img[:,:,0:3]
				origHeight, origWidth = s_img.shape[:2]
				gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
				for (x, y, w, h) in faces:
					roi_color = img[0:height, 0:width]
					s_img_Width = int(1.65*w)
					s_img_Height = int(1.65*h)
					y_ = float(s_img_Height-h)*0.6
					x_ = float(s_img_Width-w)/2
					y_1 = y-int(y_)
					x_1 = x-int(x_)
					y_2 = y_1+s_img_Height
					x_2 = x_1+s_img_Width
						
					if y_1 < 0:
						y_1 = 0
					if x_1 < 0:
						x_1 = 0
					if y_2 > height:
						y_2 = height
					if x_2 > width:
						x_2 = width

					s_img_Width = int(x_2-x_1)
					s_img_Height = int(y_2-y_1)
					s_img = cv2.resize(s_img, (s_img_Width,s_img_Height),interpolation = cv2.INTER_AREA)
					mask = cv2.resize(orig_mask, (s_img_Width,s_img_Height), interpolation = cv2.INTER_AREA)
					mask_inv = cv2.resize(orig_mask_inv, (s_img_Width,s_img_Height), interpolation = cv2.INTER_AREA)
						
					roi = roi_color[y_1:y_2, x_1:x_2]
					roi_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
					roi_fg = cv2.bitwise_and(s_img,s_img,mask = mask)
					dst = cv2.add(roi_bg,roi_fg)
					roi_color[y_1:y_2, x_1:x_2] = dst
# Tigger on face - fcflags
			if key == ord('i'):
				s_img = cv2.imread('tigger.png',-1)
				orig_mask = s_img[:,:,3]
				orig_mask_inv = cv2.bitwise_not(orig_mask)
				s_img = s_img[:,:,0:3]
				origHeight, origWidth = s_img.shape[:2]
				gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
				for (x, y, w, h) in faces:
					roi_color = img[0:height, 0:width]
					s_img_Width = int(1.3*h/origHeight*origWidth)
					s_img_Height = int(1.3*h)
					y_ = float(s_img_Height-h)*0.6
					x_ = float(s_img_Width-w)/2
					y_1 = y-int(y_)
					x_1 = x-int(x_)
					y_2 = y_1+s_img_Height
					x_2 = x_1+s_img_Width
					if y_1 < 0:
						y_1 = 0
					if x_1 < 0:
						x_1 = 0
					if y_2 > height:
						y_2 = height
					if x_2 > width:
						x_2 = width

					s_img_Width = int(x_2-x_1)
					s_img_Height = int(y_2-y_1)
					s_img = cv2.resize(s_img, (s_img_Width,s_img_Height),interpolation = cv2.INTER_AREA)
					mask = cv2.resize(orig_mask, (s_img_Width,s_img_Height), interpolation = cv2.INTER_AREA)
					mask_inv = cv2.resize(orig_mask_inv, (s_img_Width,s_img_Height), interpolation = cv2.INTER_AREA)
						
					roi = roi_color[y_1:y_2, x_1:x_2]
					roi_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
					roi_fg = cv2.bitwise_and(s_img,s_img,mask = mask)
					dst = cv2.add(roi_bg,roi_fg)
					roi_color[y_1:y_2, x_1:x_2] = dst
			# Adding stickers
			if key == ord('a'):	
				cv2.setMouseCallback('Create your own fun selfie!', self.sticker_click)
			# Adding sticker lines
			if key == ord('m'):	
				cv2.setMouseCallback('Create your own fun selfie!', self.sticker_line)
			# Return to the orignal photo
			if key == ord('t'):
				img = cv2.imread(self.filename, 1) 
				if Beautyflags == True:
					img = self.beautybrightness(img, 1.4)
			# Blur photo
			if key == ord('b'):
				img = self.smooth_img(img)
			# Brighten photo
			if key == ord('n'):
				img = self.brightness(img, 1.2)
		# Close the image window
		cv2.destroyAllWindows() 

		# Save photo
		cv2.imwrite(self.filename[:-4]+'_processed.png', img)

	def rv_draw_line(self):
		# Draw a colorful line with color changing with position
		global ix, iy, drawing
		if event == cv2.EVENT_LBUTTONDOWN:    
			ix = x
			iy = y
			drawing = True
			img[y-3:y+3,x-3:x+3] = [255-x,255-y,255-9*y]
		elif event == cv2.EVENT_MOUSEMOVE:      
			if drawing == True:
				img[y-3:y+3,x-3:x+3] = [255-x,255-y,255-9*y]
		elif event == cv2.EVENT_LBUTTONUP:     
			drawing = False
			img[y-3:y+3,x-3:x+3] = [255-x,255-y,255-9*y]


	def Video_Direction(self):
		# Directions for Video
		self.rv_top = Tk.Toplevel()  
		self.rv_top.title('Recording Video!')

		# Frame_1
		self.rv_F1 = Tk.Frame(self.rv_top,bg = "light coral")
		self.rv_F1.pack()
		# Labels for information
		self.L1 = Tk.Label(self.rv_F1, text = 'Video Direction',fg = 'brown4', bg = 'MistyRose1', font=("Courier", 30, "bold"))
		self.L1.pack()
		self.L2 = Tk.Label(self.rv_F1, text = '-----* a magic selfie-aid app that provides you with fun effects *-----',fg = 'MistyRose4', bg = 'RosyBrown1', font = ("Courier", 12, "italic"))
		self.L2.pack()
		# Frame_4
		self.rv_F4 = Tk.Frame(self.rv_top,bg = 'MistyRose1')
		self.rv_F4.pack()

		k1 = Tk.DoubleVar()
		k1.set(1.0)
		k2 = Tk.StringVar()
		k2.set(str(k1.get()))
# Directions
		direction1 = Tk.Label(self.rv_F4, font = ("Courier", 14, "bold"), text = "\n-----#####-----<   PLEASE CLICK ON START   >-----#####-----\n",fg = 'seashell4',bg = 'MistyRose1')

		direction2 = Tk.Label(self.rv_F4, text = "N----Turn your face to (left)NICK's and (right)JUDY's.",fg = 'IndianRed3', anchor="e",bg = 'MistyRose1')
		direction3 = Tk.Label(self.rv_F4, text = "I----Add mustuche under your nose.\n",fg = 'brown2', anchor="e",bg = 'MistyRose1')
		direction4 = Tk.Label(self.rv_F4, text = "M----Turn your face to (left)MICKY's and (right)MINNEY's.",fg = 'firebrick3', anchor="e",bg = 'MistyRose1')
		direction5 = Tk.Label(self.rv_F4, text = "U----Turn your face to (left)TIGGER's and (right)DONKEY's.",fg = 'red4', anchor="e",bg = 'MistyRose1')
		direction6 = Tk.Label(self.rv_F4, text = "J----Brighten.",fg = 'LightPink4', anchor="e",bg = 'MistyRose1')
		direction7 = Tk.Label(self.rv_F4, text = "K----Cancel Brighten.\n",fg = 'LightPink3', anchor="e",bg = 'MistyRose1')
		direction8 = Tk.Label(self.rv_F4, text = "V----Start Recording.",fg = 'coral4', anchor="e",bg = 'MistyRose1')
		direction9 = Tk.Label(self.rv_F4, text = "S----Pause Recording.\n",fg = 'coral3', anchor="e",bg = 'MistyRose1')
		direction10 = Tk.Label(self.rv_F4, text = "Q----Save & Quit.\n",fg = 'firebrick4', anchor="e",bg = 'MistyRose1')
		direction11 = Tk.Label(self.rv_F4, text = "###-----FILTER-----###", anchor="e",bg = 'RosyBrown1')
		direction12 = Tk.Label(self.rv_F4, text = "###-----START-----###", anchor="e",bg = 'RosyBrown1')
		direction13 = Tk.Label(self.rv_F4, text = "###-----EFFECTS-----###", anchor="e",bg = 'RosyBrown1')
		direction14 = Tk.Label(self.rv_F4, text = "###-----QUIT & SAVE-----###", anchor="e",bg = 'RosyBrown1')
		# Instructions
		direction1.pack()

		direction11.pack()
		direction6.pack()
		direction7.pack()

		direction12.pack()
		direction8.pack()
		direction9.pack()

		direction13.pack()
		direction2.pack()
		direction4.pack()
		direction5.pack()
		direction3.pack()
		
		direction14.pack()
		direction10.pack()

		self.rv_F5 = Tk.Frame(self.rv_top,bg = 'MistyRose1')
		self.rv_F5.pack()
		# Buttons
		self.B1 = Tk.Button(self.rv_F5, text = 'Start!', font = ('Courier New bold',), command = self.Video)
		self.B1.pack(side = 'left')
		self.B2 = Tk.Button(self.rv_F5, text = 'Quit!', font = ('Courier New bold',), command = self.rv_STOP)
		self.B2.pack(side = 'right')

		self.rv_F3 = Tk.Frame(self.rv_top,bg = 'MistyRose1')
		self.rv_F3.pack()
		self.L3 = Tk.Label(self.rv_F3, text = '    ',fg = 'seashell4', bg = 'white', font = ("Courier", 12, "italic"))
		self.L3.pack()

	def Video(self):
		# After choosing Start on Video mode
		cv2.namedWindow('record your video!')
		cap = cv2.VideoCapture(0)                       
		# Initial flags for functions
		videoflag = False
		brightenflag = False
		nickflag = False
		mickeyflag = False
		tiggerflag = False
		noseflag = False
		drawlineflag = False


		width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)       
		height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) 
		height = int(2*height/3)
		width = int(2*width/3) 
		shape = (width, height)               
		fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
		FPS = 20.0       
		# Create output video

		# output_path = "./"
		ts = datetime.datetime.now() # grab the current timestamp
		self.video_filename = "{}.avi".format(ts.strftime("%Y-%m-%d_%H-%M-%S"))  # construct filename
		# p = os.path.join(output_path, self.filename)  # construct output path
		# self.current_image.save(p, "PNG")  # save image as jpeg file
		# print("[INFO] saved {}".format(self.filename))
		# self.IMG_PROCESSING()


		out = cv2.VideoWriter(self.video_filename, fourcc, FPS, shape)
		
		while cap.isOpened():

			ok, img = cap.read()
			img = cv2.resize(img,(int(width), int(height)), interpolation = cv2.INTER_CUBIC)
			if ok == True:
				if brightenflag == True:
					img = self.brightness(img, 1.4)
				if videoflag==True:
					if nickflag == False and mickeyflag == False and noseflag == False and tiggerflag == False:
						out.write(img)
				
				if nickflag==True:
					faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
					gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
					faces = faceCascade.detectMultiScale(gray,scaleFactor=1.15,minNeighbors=5,minSize=(30, 30),flags = cv2.CASCADE_SCALE_IMAGE)
					for (x, y, w, h) in faces:
						if x+w/2 < width/2:
							s_img = cv2.imread('nick.png',-1)
							orig_mask = s_img[:,:,3]
							orig_mask_inv = cv2.bitwise_not(orig_mask)
							s_img = s_img[:,:,0:3]
							origHeight, origWidth = s_img.shape[:2]
							roi_color = img[0:height, 0:width]
							s_img_Width = int(1.65*w)
							s_img_Height = int(1.65*h)
							y_ = float(s_img_Height-h)*0.6
							x_ = float(s_img_Width-w)/2
							y_1 = y-int(y_)
							x_1 = x-int(x_)
							y_2 = y_1+s_img_Height
							x_2 = x_1+s_img_Width
							if y_1 < 0:
								y_1 = 0
							if x_1 < 0:
								x_1 = 0
							if y_2 > height:
								y_2 = height
							if x_2 > width:
								x_2 = width
							s_img_Width = int(x_2-x_1)
							s_img_Height = int(y_2-y_1)
							s_img = cv2.resize(s_img, (s_img_Width,s_img_Height),interpolation = cv2.INTER_AREA)
							mask = cv2.resize(orig_mask, (s_img_Width,s_img_Height), interpolation = cv2.INTER_AREA)
							mask_inv = cv2.resize(orig_mask_inv, (s_img_Width,s_img_Height), interpolation = cv2.INTER_AREA)
							roi = roi_color[y_1:y_2, x_1:x_2]
							roi_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
							roi_fg = cv2.bitwise_and(s_img,s_img,mask = mask)
							dst = cv2.add(roi_bg,roi_fg)
							roi_color[y_1:y_2, x_1:x_2] = dst
						else:
							s_img = cv2.imread('judy.png',-1)
							orig_mask = s_img[:,:,3]
							orig_mask_inv = cv2.bitwise_not(orig_mask)
							s_img = s_img[:,:,0:3]
							origHeight, origWidth = s_img.shape[:2]
							roi_color = img[0:height, 0:width]
							s_img_Width = int(1.55*h/origHeight*origWidth)
							s_img_Height = int(1.55*h)
							y_ = float(s_img_Height-h)*1.0
							x_ = float(s_img_Width-w)/2
							y_1 = y-int(y_)
							x_1 = x-int(x_)
							y_2 = y_1+s_img_Height
							x_2 = x_1+s_img_Width
							if y_1 < 0:
								y_1 = 0
							if x_1 < 0:
								x_1 = 0
							if y_2 > height:
								y_2 = height
							if x_2 > width:
								x_2 = width
							s_img_Width = int(x_2-x_1)
							s_img_Height = int(y_2-y_1)
							s_img = cv2.resize(s_img, (s_img_Width,s_img_Height),interpolation = cv2.INTER_AREA)
							mask = cv2.resize(orig_mask, (s_img_Width,s_img_Height), interpolation = cv2.INTER_AREA)
							mask_inv = cv2.resize(orig_mask_inv, (s_img_Width,s_img_Height), interpolation = cv2.INTER_AREA)
							roi = roi_color[y_1:y_2, x_1:x_2]
							roi_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
							roi_fg = cv2.bitwise_and(s_img,s_img,mask = mask)
							dst = cv2.add(roi_bg,roi_fg)
							roi_color[y_1:y_2, x_1:x_2] = dst
					if videoflag==True:
							out.write(img)

				if mickeyflag==True:

					faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
					gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
					faces = faceCascade.detectMultiScale(gray,scaleFactor=1.15,minNeighbors=5,minSize=(30, 30),flags = cv2.CASCADE_SCALE_IMAGE)
					for (x, y, w, h) in faces:
						if x+w/2 < width/2:
							s_img = cv2.imread('mickey.png',-1)
							orig_mask = s_img[:,:,3]
							orig_mask_inv = cv2.bitwise_not(orig_mask)
							s_img = s_img[:,:,0:3]
							origHeight, origWidth = s_img.shape[:2]
							roi_color = img[0:height, 0:width]
							s_img_Width = int(1.7*w)
							s_img_Height = int(1.7*h)
							y_ = float(s_img_Height-h)*0.96
							x_ = float(s_img_Width-w)/2
							y_1 = y-int(y_)
							x_1 = x-int(x_)
							y_2 = y_1+s_img_Height
							x_2 = x_1+s_img_Width
							if y_1 < 0:
								y_1 = 0
							if x_1 < 0:
								x_1 = 0
							if y_2 > height:
								y_2 = height
							if x_2 > width:
								x_2 = width
							s_img_Width = int(x_2-x_1)
							s_img_Height = int(y_2-y_1)
							s_img = cv2.resize(s_img, (s_img_Width,s_img_Height),interpolation = cv2.INTER_AREA)
							mask = cv2.resize(orig_mask, (s_img_Width,s_img_Height), interpolation = cv2.INTER_AREA)
							mask_inv = cv2.resize(orig_mask_inv, (s_img_Width,s_img_Height), interpolation = cv2.INTER_AREA)
							roi = roi_color[y_1:y_2, x_1:x_2]
							roi_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
							roi_fg = cv2.bitwise_and(s_img,s_img,mask = mask)
							dst = cv2.add(roi_bg,roi_fg)
							roi_color[y_1:y_2, x_1:x_2] = dst
						else:
							s_img = cv2.imread('miny.png',-1)
							orig_mask = s_img[:,:,3]
							orig_mask_inv = cv2.bitwise_not(orig_mask)
							s_img = s_img[:,:,0:3]
							origHeight, origWidth = s_img.shape[:2]
							roi_color = img[0:height, 0:width]
							s_img_Width = int(1.7*w)
							s_img_Height = int(1.7*h)
							y_ = float(s_img_Height-h)*0.96
							x_ = float(s_img_Width-w)/2
							y_1 = y-int(y_)
							x_1 = x-int(x_)
							y_2 = y_1+s_img_Height
							x_2 = x_1+s_img_Width
							if y_1 < 0:
								y_1 = 0
							if x_1 < 0:
								x_1 = 0
							if y_2 > height:
								y_2 = height
							if x_2 > width:
								x_2 = width
							s_img_Width = int(x_2-x_1)
							s_img_Height = int(y_2-y_1)
							s_img = cv2.resize(s_img, (s_img_Width,s_img_Height),interpolation = cv2.INTER_AREA)
							mask = cv2.resize(orig_mask, (s_img_Width,s_img_Height), interpolation = cv2.INTER_AREA)
							mask_inv = cv2.resize(orig_mask_inv, (s_img_Width,s_img_Height), interpolation = cv2.INTER_AREA)
							roi = roi_color[y_1:y_2, x_1:x_2]
							roi_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
							roi_fg = cv2.bitwise_and(s_img,s_img,mask = mask)
							dst = cv2.add(roi_bg,roi_fg)
							roi_color[y_1:y_2, x_1:x_2] = dst
					if videoflag==True:
							out.write(img)

				if tiggerflag==True:

					faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
					gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
					faces = faceCascade.detectMultiScale(gray,scaleFactor=1.15,minNeighbors=5,minSize=(30, 30),flags = cv2.CASCADE_SCALE_IMAGE)
					for (x, y, w, h) in faces:
						if x+w/2 < width/2:
							s_img = cv2.imread('tigger.png',-1)
							orig_mask = s_img[:,:,3]
							orig_mask_inv = cv2.bitwise_not(orig_mask)
							s_img = s_img[:,:,0:3]
							origHeight, origWidth = s_img.shape[:2]
							roi_color = img[0:height, 0:width]
							s_img_Width = int(1.3*h/origHeight*origWidth)
							s_img_Height = int(1.3*h)
							y_ = float(s_img_Height-h)*0.6
							x_ = float(s_img_Width-w)/2
							y_1 = y-int(y_)
							x_1 = x-int(x_)
							y_2 = y_1+s_img_Height
							x_2 = x_1+s_img_Width
							if y_1 < 0:
								y_1 = 0
							if x_1 < 0:
								x_1 = 0
							if y_2 > height:
								y_2 = height
							if x_2 > width:
								x_2 = width
							s_img_Width = int(x_2-x_1)
							s_img_Height = int(y_2-y_1)
							s_img = cv2.resize(s_img, (s_img_Width,s_img_Height),interpolation = cv2.INTER_AREA)
							mask = cv2.resize(orig_mask, (s_img_Width,s_img_Height), interpolation = cv2.INTER_AREA)
							mask_inv = cv2.resize(orig_mask_inv, (s_img_Width,s_img_Height), interpolation = cv2.INTER_AREA)
							roi = roi_color[y_1:y_2, x_1:x_2]
							roi_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
							roi_fg = cv2.bitwise_and(s_img,s_img,mask = mask)
							dst = cv2.add(roi_bg,roi_fg)
							roi_color[y_1:y_2, x_1:x_2] = dst
						else:
							s_img = cv2.imread('donkey.png',-1)
							orig_mask = s_img[:,:,3]
							orig_mask_inv = cv2.bitwise_not(orig_mask)
							s_img = s_img[:,:,0:3]
							origHeight, origWidth = s_img.shape[:2]
							roi_color = img[0:height, 0:width]
							s_img_Width = int(1.3*h/origHeight*origWidth)
							s_img_Height = int(1.3*h)
							y_ = float(s_img_Height-h)*1.0
							x_ = float(s_img_Width-w)/2
							y_1 = y-int(y_)
							x_1 = x-int(x_)
							y_2 = y_1+s_img_Height
							x_2 = x_1+s_img_Width
							if y_1 < 0:
								y_1 = 0
							if x_1 < 0:
								x_1 = 0
							if y_2 > height:
								y_2 = height
							if x_2 > width:
								x_2 = width
							s_img_Width = int(x_2-x_1)
							s_img_Height = int(y_2-y_1)
							s_img = cv2.resize(s_img, (s_img_Width,s_img_Height),interpolation = cv2.INTER_AREA)
							mask = cv2.resize(orig_mask, (s_img_Width,s_img_Height), interpolation = cv2.INTER_AREA)
							mask_inv = cv2.resize(orig_mask_inv, (s_img_Width,s_img_Height), interpolation = cv2.INTER_AREA)
							roi = roi_color[y_1:y_2, x_1:x_2]
							roi_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
							roi_fg = cv2.bitwise_and(s_img,s_img,mask = mask)
							dst = cv2.add(roi_bg,roi_fg)
							roi_color[y_1:y_2, x_1:x_2] = dst
					if videoflag==True:
							out.write(img)

				if noseflag == True:

					faceCascadeFilePath = "haarcascade_frontalface_default.xml"
					noseCascadeFilePath = "haarcascade_mcs_nose.xml"
					faceCascade = cv2.CascadeClassifier(faceCascadeFilePath)
					noseCascade = cv2.CascadeClassifier(noseCascadeFilePath) 
					imgMustache = cv2.imread('mustache.png',-1)
					orig_mask = imgMustache[:,:,3]
					orig_mask_inv = cv2.bitwise_not(orig_mask)
					imgMustache = imgMustache[:,:,0:3]
					origMustacheHeight, origMustacheWidth = imgMustache.shape[:2]
					gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
					faces = faceCascade.detectMultiScale(
							gray,
							scaleFactor=1.1,
							minNeighbors=5,
							minSize=(30, 30),
							flags = cv2.CASCADE_SCALE_IMAGE
					)
					for (x, y, w, h) in faces:
						roi_gray = gray[y:y+h, x:x+w]
						roi_color = img[y:y+h, x:x+w]
						nose = noseCascade.detectMultiScale(roi_gray)
						for (nx,ny,nw,nh) in nose:
							mustacheWidth =  3 * nw
							mustacheHeight = mustacheWidth * origMustacheHeight / origMustacheWidth
							x1 = nx - int(mustacheWidth/4)
							x2 = nx + nw + int(mustacheWidth/4)
							y1 = ny + nh - int(mustacheHeight/2)
							y2 = ny + nh + int(mustacheHeight/2)
            				# Check for clipping
							if x1 < 0:
								x1 = 0
							if y1 < 0:
								y1 = 0
							if x2 > w:
								x2 = w
							if y2 > h:
								y2 = h
							mustacheWidth = x2 - x1
							mustacheHeight = y2 - y1
							mustache = cv2.resize(imgMustache, (mustacheWidth,mustacheHeight), interpolation = cv2.INTER_AREA)
							mask = cv2.resize(orig_mask, (mustacheWidth,mustacheHeight), interpolation = cv2.INTER_AREA)
							mask_inv = cv2.resize(orig_mask_inv, (mustacheWidth,mustacheHeight), interpolation = cv2.INTER_AREA)
							roi = roi_color[y1:y2, x1:x2]
							roi_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
							roi_fg = cv2.bitwise_and(mustache,mustache,mask = mask)
							dst = cv2.add(roi_bg,roi_fg)
							roi_color[y1:y2, x1:x2] = dst
					

						if videoflag==True:
							# Outputing video
							out.write(img)

				cv2.imshow('Live video',img)
				key = cv2.waitKey(1)

				if key == ord('v'):
					videoflag = True
					out.write(img)
					print('recording video')

				if key == ord('s'):
					videoflag = False
					print('stop recording')
				if key == ord('j'):
					brightenflag = True
				if key == ord('k'):
					brightenflag = False
				
				if key == ord('n'):
					nickflag = True
					tiggerflag = False
					noseflag = False
					mickeyflag = False

				if key == ord('i'):	
					noseflag = True
					
					nickflag = False
					mickeyflag = False
					tiggerflag = False

				if key == ord('m'):	
					mickeyflag = True

					nickflag = False
					tiggerflag = False
					noseflag = False

				if key == ord('u'):	
					tiggerflag = True

					nickflag = False
					mickeyflag = False
					noseflag = False

				if key == ord('q'):
					cap.release()
					out.release()
					cv2.destroyAllWindows()

			else:
				break





# Start the total program
DSP_PROJ()









