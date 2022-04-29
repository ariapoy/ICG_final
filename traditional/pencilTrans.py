import os, sys
import cv2
import numpy as np
from scipy import ndimage
from scipy.sparse.linalg import cg
from skimage.transform import rotate
from scipy.sparse import spdiags, dia_matrix
import argparse

def gen_pencil(im, P, J):
    ## Parameters
    theta = 0.2

    H, W = im.shape

    ## Initialization
    P = cv2.resize(P, (W, H))
    logP = np.log(P.ravel())
    logP = spdiags(logP, 0, H*W, H*W)

    J = cv2.resize(J, (W, H))
    logJ = np.log(J.ravel()).reshape(-1, 1)

    e = np.ones(H*W)
    Dx = spdiags([-e,e], [0, H*W], H*W, H*W)
    Dy = spdiags([-e,e], [0, 1], H*W, H*W)

    ## Compute matrix A and b
    A = (Dx.dot(Dx.T) + Dy.dot(Dy.T)).multiply(theta) + (logP.T).dot(logP)
    b = (logP.T).dot(logJ)

    ## Conjugate gradient
    beta, _ = cg(A, b, tol=1e-6, maxiter=60)

    ## Compute the result
    beta = beta.reshape((H, W))

    #print(P.max(), P.min(), J.max(), J.min())
    #P = P.reshape((H, W))

    T = np.power(P, beta)

    return T

def gen_stroke(im, ks, width, dirNum, sks):

    ## Initialization
    H, W = im.shape

    ## Smoothing
    im = np.clip(im*255, 0, 255).astype(np.uint8)
    im = ndimage.median_filter(im, size=sks, mode='constant', cval=0)
    im = im.astype(float)/255.

    ## Image gradient
    imX = np.hstack([abs(im[:, :-1] - im[:, 1:]), np.zeros((H,1))])
    imY = np.vstack([abs(im[:-1, :] - im[1:, :]), np.zeros((1,W))])
    imEdge = imX + imY

    ## Convolution kernel with horizontal direction
    kerRef = np.zeros((ks*2+1, ks*2+1))
    kerRef[ks, :] = 1

    ## Classification
    response = np.zeros((H, W, dirNum))
    for n in range(dirNum):
        ker = rotate(kerRef, (n-1)*180/dirNum, order=1)
        response[:, :, n] = cv2.filter2D(imEdge, -1, ker)

    idx = response.argmax(axis=-1)
    #[~, index] = max(response, [], 3)

    ## Create the stroke
    C = np.zeros((H, W, dirNum))
    for n in range(dirNum):
        C[:, :, n] = imEdge * (idx == n)

    kerRef = np.zeros((ks*2+1, ks*2+1))
    kerRef[ks, :] = 1
    for n in range(width):
        if (ks - n) >= 0:
            kerRef[ks - n,:] = 1
        if (ks + n) <= (ks*2):
            kerRef[ks + n,:] = 1

    Spn = np.zeros((H, W, dirNum))
    for n in range(dirNum):
        ker = rotate(kerRef, (n-1)*180/dirNum, order=1)
        Spn[:, :, n] = cv2.filter2D(C[:, :, n], -1, ker)

    Sp = Spn.sum(axis=-1)
    Sp = (Sp - Sp.min()) / (Sp.max() - Sp.min())
    S = 1 - Sp

    return S


def hist_match(source, tHist):

    oldshape = source.shape
    source = source.ravel()

    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True, return_counts=True)
    t_values = np.arange(256)
    #t_values, t_counts = np.unique(template, return_counts=True)

    s_quantiles = np.cumsum(s_counts).astype(float)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(tHist).astype(float)
    #t_quantiles /= t_quantiles[-1]

    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(oldshape)


def gen_tone_map(img, grp):
    ## Parameters
    Ub = 225
    Ua = 105
    Mud = 90
    DeltaB = 9
    DeltaD = 11

    # groups from dark to light
    if grp == 0:
        ## 1st group
        Omega3 = 42
        Omega2 = 29
        Omega1 = 29
    elif grp == 1:
        # 2nd group
        Omega3 = 52
        Omega2 = 37
        Omega1 = 11
    elif grp == 2:
        # 3rd group
        Omega3 = 76
        Omega2 = 22
        Omega1 = 2
    else:
        raise

    # Compute the target histgram
    histgramTarget = np.zeros(256)
    total = 0
    for i in range(256):
        if i < Ua or i > Ub:
            p = 0
        else:
            p = 1 / (Ub - Ua)

        histgramTarget[i] = (\
                Omega1 * 1/DeltaB * np.exp(-(255 - i) / DeltaB) + \
                    Omega2 * p + \
                    Omega3 * 1/np.sqrt(2 * np.pi * DeltaD) * np.exp(-(i - Mud)**2 / (2 * DeltaD**2))) * 0.01

        total = total + histgramTarget[i]
    histgramTarget = histgramTarget/total

    img = np.clip(img*255, 0, 255).astype(np.uint8)

    ## Histgram matching
    J = hist_match(img, histgramTarget)

    ## Smoothing
    J = cv2.blur(J, (10, 10))

    return J.astype(float)/255.

def pencilTrans(im_path, ks, swidth, dirNum, sks, gammaS, gammaI, grp, pencil_type, is_rgb):
    if is_rgb:
        img = cv2.imread(im_path, 1)
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        img_lum = img_yuv[:, :, 0]
    else:
        img = cv2.imread(im_path, 0)
        img_lum = img.copy()
    #cv2.imshow('0', img)

    ## trans img into [0, 1]
    img_lum = img_lum.astype(float)/255.

    S = gen_stroke(img_lum, ks, swidth, dirNum, sks)
    S = S**gammaS
    #cv2.imshow('1', (S*255).astype('uint8'))

    ## Generate the tone map
    J = gen_tone_map(img_lum, grp)
    J = J**gammaI
    #cv2.imshow('2', (J*255).astype('uint8'))

    ## Read the pencil texture
    P = cv2.imread('pencils/pencil%d.jpg'%pencil_type, 0)
    P = P.astype(float)/255.

    ## Generate the pencil map
    T = gen_pencil(img_lum, P, J)
    #cv2.imshow('3', (T*255).astype('uint8'))

    ## Compute the result
    img_lum = S*T

    ## trans img into [0, 1]
    img_lum = np.clip(img_lum*255, 0, 255).astype(np.uint8)
    #cv2.imshow('4', img_lum)

    if is_rgb:
        img_yuv[:, :, 0] = img_lum.copy()
        I = cv2.cvtColor(img_yuv, cv2.COLOR_YCrCb2BGR)
    else:
        I = img_lum

    return I

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_path', dest='input_path',
                      help='the file path of the original image)',
                      required=True, type=str)
    parser.add_argument('--save_path', dest='save_path',
                      help='the path for saving result)',
                      required=True, type=str)
    parser.add_argument('-ks', dest='ks',
                      help='size of the line segement kernel (usually 1/30 of the height/width of the original image)',
                      default=8, type=int)
    parser.add_argument('-sw', dest='sw',
                      help='thickness of the strokes in the Stroke Map (0, 1, 2)',
                      default=1, type=int)
    parser.add_argument('-nd', dest='nd',
                      help='stroke directions in the Stroke Map (used for the kernels)',
                      default=8, type=int)
    parser.add_argument('-sks', dest='sks',
                      help='the size of smoothed kernel',
                      default=3, type=int)
    parser.add_argument('-wg', dest='wg',
                      help='3 possible weight groups (0, 1, 2) for the histogram distribution, according to the paper (brighter to darker)',
                      default=0, choices={0, 1, 2}, type=int)
    parser.add_argument('-pt', dest='pt',
                      help='5 possible pencil types',
                      default=1, choices={0, 1, 2, 3, 4}, type=int)
    parser.add_argument('-sd', dest='sd',
                      help='stroke_darkness: 1 is the same, up is darker',
                      default=1, type=float)
    parser.add_argument('-td', dest='td',
                      help='tone_darkness: 1 is the same, up is darker',
                      default=1, type=float)
    parser.add_argument('--rgb', dest='rgb',
                      help='True if the original image has 3 channels, False if grayscale',
                      action='store_true')
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()
    im = pencilTrans(args.input_path, args.ks, args.sw, args.nd, args.sks, args.sd, args.td, args.wg, args.pt, args.rgb)
    cv2.imwrite(args.save_path, im)
    #cv2.imshow('5', im)
    #cv2.waitKey(0)
