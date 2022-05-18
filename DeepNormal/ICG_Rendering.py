import cv2
import os
from tqdm import tqdm
import numpy as np 
import math
import sys
import argparse
from utils import *

import pdb

#Some function definition for image based rendering
def GuidedFiltF(img, r):
    eps = 0.04;
    I = img
    I2 = cv2.pow(I,2);
    mean_I = cv2.boxFilter(I,-1,((2*r)+1,(2*r)+1))
    mean_I2 = cv2.boxFilter(I2,-1,((2*r)+1,(2*r)+1))
    
    cov_I = mean_I2 - cv2.pow(mean_I,2);
    
    var_I = cov_I;
    
    a = cv2.divide(cov_I,var_I+eps)
    b = mean_I - (a*mean_I)
    
    mean_a = cv2.boxFilter(a,-1,((2*r)+1,(2*r)+1))
    mean_b = cv2.boxFilter(b,-1,((2*r)+1,(2*r)+1))
    
    q = (mean_a * I) + mean_b;
    return(q)

def ComputeLightDirectionMat(Xpos, Ypos, Zpos, IndexMat3D):
	out = np.copy(IndexMat3D)
	Z = IndexMat3D[:,:,0] + Zpos
	Y = IndexMat3D[:,:,1] - Ypos
	X = Xpos - IndexMat3D[:,:,2]

	SUM = np.sqrt(X**2 + Y**2 + Z**2)

	out[:,:,0] = Z / SUM
	out[:,:,1] = Y / SUM
	out[:,:,2] = X / SUM

	return out


def CreateIndexMat(height, width):
	ind = np.zeros((height, width, 3))
	for j in range(0, height):
		for i in range (0, width):
			ind[j,i,0] = 0
			ind[j,i,1] = j
			ind[j,i,2] = i
	return ind

def ComputeFresnel(dot, ior):
	height, width = dot.shape
	cosi = np.copy(dot)
	etai = np.ones((height, width))
	#etat = np.ones((height, width)) * ior
	etat = ior
	# Snell's law
	sint = etai/etat * np.sqrt(np.maximum(0.0, cosi * cosi))
	#Total Reflection
	sint2= np.copy(sint)
	#sint[np.where(sint >=1)] = 1

	cost = np.sqrt(np.maximum(0.0, 1 - sint * sint))
	cosi = abs(cosi)
	sint = (((etat * cosi) - (etai * cost)) / ((etat * cosi) + (etai * cost))**2 + ((etai * cosi) - (etat * cost)) / ((etai * cosi) + (etat * cost))**2)/2.0
	sint[np.where(sint2 >=1)] = 1
	#Rs = ((etat * cosi) - (etai * cost)) / ((etat * cosi) + (etai * cost))
	#Rp = ((etai * cosi) - (etat * cost)) / ((etai * cosi) + (etat * cost))
	#kr = (Rs * Rs + Rp * Rp) / 2

	return  1-sint

parser = argparse.ArgumentParser(description='')
parser.add_argument('--docker_path', dest='P', default='./', help='Path of shared docker directory, default is ./ for using without docker')
parser.add_argument('--lineart_path', dest='lineart_path', default='Pepper/Lines.jpg', help='Path of the linedrawing (grayscale lines)')
parser.add_argument('--mask_path', dest='mask_path', default='Pepper/Mask.jpg', help='Path of the mask')
parser.add_argument('--normal_path', dest='normal_path', default='RES/Normal_Map.png', help='Path of the normal map')
parser.add_argument('--color_path', dest='color_path', default='Pepper/Colors.jpg', help='Path of the flat colors')
parser.add_argument('--r', dest='r', type= float, default=0.99, help='r value for light')
parser.add_argument('--g', dest='g', type= float, default=0.83, help='r value for light')
parser.add_argument('--b', dest='b', type= float, default=0.66, help='r value for light')
parser.add_argument('--save_path', dest ='save_path', default='RES/', help='Path of the save folder')
args = parser.parse_args()

# Add docker prefix if needed
args.lineart_path = args.P + args.lineart_path
args.normal_path = args.P + args.normal_path
args.mask_path = args.P + args.mask_path
args.color_path = args.P + args.color_path
args.save_path = args.P + args.save_path

#Check if images exist
if not os.path.isfile(args.lineart_path):
	sys.exit("Error, couldn't read lineart image, file doesn't exist")
if not os.path.isfile(args.normal_path):
	sys.exit("Error, couldn't read normal map image, file doesn't exist")
if not os.path.isfile(args.mask_path):
	sys.exit("Error, couldn't read mask, file doesn't exist")
if not os.path.isfile(args.color_path):
	sys.exit("Error, couldn't read flat colors image, file doesn't exist")

Light_direction = Normalize([0,0,1]) #update realtime later


#Load Lineart
img = load_image(args.lineart_path)
#Load Normal Map
imgN = load_normal(args.normal_path)
#Load Mask
Mask = load_mask(args.mask_path)
#Load Color
color = load_color(args.color_path)
print(" ")

height, width = img.shape

imgN = imgN / 127.5 - 1.0

#Some Init
pi = math.pi
threshold = 100
Xpos = 0
Ypos = 0
Zpos = 100
filtering = 0
amb = 0.55
ks = 0
alpha =10
num = 0
ind = CreateIndexMat(height, width)
r = args.r
g = args.g
b = args.b
Plight = 0.8

#print(ind)
loop = False
t = 0
# while loop:
while t <= 9:

	if(filtering >0):
		imgN2 = GuidedFiltF(imgN, filtering)
	else :
		imgN2 = np.copy(imgN)

	LD = ComputeLightDirectionMat(Xpos, Ypos, Zpos, ind)

	dot = np.sum(imgN2 * LD, axis = 2)

	dot[np.where(dot<0)]=0
	dot[np.where(dot>1.0)]=1.0

	dot3 = np.stack((dot,dot,dot), axis = 2)
	R = (np.multiply(2*dot3,imgN2) - LD)[:,:,0]
	R[np.where(R<0)]=0


	Rspec = (R**alpha)
	RspecR = (R**(50.0 * alpha/10.0))
	RspecG = (R**(50.0 * alpha/10.0))
	RspecB = (R**(53.47* alpha/10.0))

	#Schlik
	FresnelR = RspecR + (1-RspecR) * (1.0-R)**5
	FresnelG = RspecG + (1-RspecG) * (1.0-R)**5
	FresnelB = RspecB + (1-RspecB) * (1.0-R)**5

	dstImage = dot

	dot8 = (dot*255).astype(np.dtype('uint8'))
	color64 = color.astype(np.dtype('float64'))

	if len(color64.shape) < 3:
		# print("This is grayscale image.\nLet's convert it to three channels.")
		color64 = np.stack((color64,)*3, axis=-1)

	color64[:,:,0] = np.minimum(255.0, color64[:,:,0] * amb * b + Plight * color64[:,:,0] * dstImage * b + Plight * b * 1.58*ks * RspecB*FresnelB)
	color64[:,:,1] = np.minimum(255.0, color64[:,:,1] * amb * g + Plight * color64[:,:,1] * dstImage * g + Plight * g * 1.50*ks * RspecG*FresnelG)
	color64[:,:,2] = np.minimum(255.0, color64[:,:,2] * amb * r + Plight * color64[:,:,2] * dstImage * r + Plight * r * 1.35*ks * RspecR*FresnelR)

	color64[np.where(Mask == 0)]= 255
	final = color64.astype(np.dtype('uint8'))
	# cv2.imshow('final', final)
	cv2.imwrite('test{0}.png'.format(t), final)
	t += 1

	pos_diff = 100 * 8
	if t == 1:
		Xpos += pos_diff
		r = 0.01
		g = 0.83
		b = 0.66
	elif t == 2:
		Xpos -= pos_diff
		r = 0.99
		Ypos -= pos_diff
	elif t == 3:
		Ypos += pos_diff
		Zpos += pos_diff
	elif t == 4:
		Zpos -= pos_diff
		filtering += 1
	elif t == 5:
		filtering -= 1
		amb += 0.05
	elif t == 6:
		amb -= 0.05
		ks -= 50
	elif t == 7:
		ks += 50
		alpha += 2
	elif t == 8:
		alpha -= 2
		Xpos += pos_diff
		Xpos += pos_diff
		Zpos += pos_diff
	elif t == 9:
		Xpos -= pos_diff
		Xpos -= pos_diff
		Zpos -= pos_diff
		r = 0.70
		g = 0.70
		b = 0.70