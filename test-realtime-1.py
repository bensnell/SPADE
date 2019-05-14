"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os
from collections import OrderedDict

import data
from options.test_options import TestOptions
from models.pix2pix_model import Pix2PixModel
from util.visualizer import Visualizer
from util import html

import numpy as np
import matplotlib.pyplot as plt

import time

import cv2
import win32gui

import torch

# Global variables
mousePressed = False
mousePressedLocation = (0,0)
mousePressedOffset = [0,0]

lastMouseLocation = [0,0]

# These semantics are one more than the actual one
semantics = [[157,"sky"],[106,"clouds"],[155,"sea"],[159,"snow"],[111,"dirt"],[135,"mountain"],[150,"rock"],[169,"tree"],[124,"grass"],[128,"house"],[158,"skyscraper"],[154,"sand"]]

# Mouse Pressed Function
def onpress(event):
	global mousePressed, mousePressedLocation, mousePressedOffset
	button=['left','middle','right']
	toolbar=plt.get_current_fig_manager().toolbar
	if toolbar.mode != '' or event.xdata == None or event.ydata == None:
		print("Toolbar in mode {:s}.".format(toolbar.mode))
	else:
		# The xdata and ydata are the image coords
		# The x and y are the figure (subwindow) coords
		mousePressed = True
		mousePressedLocation = win32gui.GetCursorPos();
		mousePressedOffset = [event.xdata, event.ydata]
		# print("You {0}-pressed coords ({1},{2}) (pix ({3},{4}))".format(button[event.button+1],event.xdata,event.ydata,event.x,event.y))

# Mouse Released Functio
def onrelease(event):
	global mousePressed
	global mousePressedLocation
	button=['left','middle','right']
	toolbar=plt.get_current_fig_manager().toolbar
	if toolbar.mode != '':
		print("Toolbar in mode {:s}.".format(toolbar.mode))
	else:
		# The xdata and ydata are the image coords
		# The x and y are the figure (subwindow) coords
		mousePressed = False
		# print("You {0}-released coords ({1},{2}) (pix ({3},{4}))".format(button[event.button+1],event.xdata,event.ydata,event.x,event.y))

def mousemove(event):
	global lastMouseLocation
	x,y = event.xdata, event.ydata
	if x != None and y != None:
		lastMouseLocation = [x, y]

brushSize = 20

def scrollevent(event):
	global brushSize
	if event.button == 'up':
		brushSize = max(brushSize - 10, 5)
	else:
		brushSize = min(brushSize + 10, 300)

selectedSemantic = 0

def onkey(event):
	global selectedSemantic, semantics
	try:
		val = int(event.key)
		if val >= 0 and val <= 10:
			selectedSemantic = val
			print("Selected {:s}".format(semantics[selectedSemantic][1]))
	except:
		None


def main():

	global mousePressed, mousePressedLocation, mousePressedOffset, lastMouseLocation, brushSize, selectedSemantic, semantics

	# Create a figure to draw to and connect the mouse functions
	fig = plt.figure()
	fig.canvas.mpl_connect('button_press_event', onpress)
	fig.canvas.mpl_connect('button_release_event', onrelease)
	fig.canvas.mpl_connect('motion_notify_event', mousemove)
	fig.canvas.mpl_connect('scroll_event', scrollevent)
	fig.canvas.mpl_connect('key_press_event', onkey)
	# Create the image that is drawn to the figure
	imgData=np.zeros((256, 256))
	img = plt.imshow(imgData);
	plt.pause(0.05)

	# First, load the model
	opt = TestOptions().parse()
	dataloader = data.create_dataloader(opt)
	model = Pix2PixModel(opt)
	model.eval()
	visualizer = Visualizer(opt)

	# Create a webpage that summarizes the all results
	web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
	webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

	# Get the format of the input data
	inData = None
	for i, data_i in enumerate(dataloader):
		inData = data_i
		break
	# Initially, Set the input to zero
	inData['image'][:,:,:,:] = 0.0;
	# initially, set the labels to zero
	inData['label'][0,0,:,:] = 0

	tmpImg = np.zeros((1,1,256,256), dtype=np.uint8);

	# While the program is running, generate imagery from input
	while True:

		# Update the mouse location
		if mousePressed:

			# Doesn't work:
			# Get current location
			# cursorPos = win32gui.GetCursorPos()
			# mouseLocation = [cursorPos[0] - mousePressedLocation[0] + mousePressedOffset[0], cursorPos[1] - mousePressedLocation[1] + mousePressedOffset[1]]
			
			# Print the last mouse location
			# print(lastMouseLocation)

			# Draw any mouse movements to the input image
			# inData['label'][0,0] = 0

			cv2.circle(tmpImg[0,0], (int(lastMouseLocation[0]),int(lastMouseLocation[1])), brushSize, (semantics[selectedSemantic][0]-1), -1)
			inData['label'] = torch.tensor(tmpImg)

			# data_i['label']

		# Run a forward pass through the model to "infer" the output image
		generated = model(inData, mode='inference')

		img_path = inData['path']
		for b in range(generated.shape[0]):

			# Extract the visuals
			# print('process image... %s' % img_path[b])
			visuals = OrderedDict([('input_label', data_i['label'][b]),('synthesized_image', generated[b])])

			# Draw the image 
			imgData = visualizer.convert_visuals_to_numpy(visuals)['synthesized_image'];

			# Draw the cursor location and size
			cv2.circle(imgData, (int(lastMouseLocation[0]),int(lastMouseLocation[1])), brushSize, (255, 0, 0), 2)
			img.set_data(imgData);

			# img.set_data(tmpImg[0,0])

			plt.pause(0.05)

# webpage.save()





if __name__ == "__main__":
	main()