import numpy as np #for array calculations
import argparse #for terminal argument input
from PIL import Image , ImageTk, ImageCms # image manipulation
from scipy import ndimage, signal 
from skimage.measure import compare_ssim as ssim #ssim calculations
import math
from skimage import restoration #to compare my code with module
import cv2 #image manipulation
## calculating psnr taken from internet from this repo - https://github.com/aizvorski/video-quality
def psnr(img1, img2):
	mse = np.mean( (img1 - img2) ** 2 )
	if mse == 0:
		return 100
	PIXEL_MAX = 255.0
	return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

# ## Compute DFT function written by me
def dft2(img):
	# find shape of array, store them in M,N
	M = img.shape[0]
	N = img.shape[1]
	fm = np.zeros((M,M),dtype=complex) #create fourier matrix of size M
	fm = np.matrix(fm)
	for k in range(M):
		for m in range(M):
			fm[k,m] = np.exp((-2.0*1j*np.pi*m*k)/M)
	fn = np.zeros((N,N),dtype=complex) # create fourier matrix of size N
	fn = np.matrix(fn)
	for l in range(N):
		for n in range(N):
			fn[l,n] = np.exp((-2.0*1j*np.pi*n*l)/N)
	img = np.matrix(img)
	dftimg = np.matmul(np.matmul(fm,img),fn) # calculate DFT by matrix multiplication of fourier matrices with img array
	return dftimg
## Compute IDFT written by me
def idft2(img):
	# find shape of array, store them in M,N
	M = img.shape[0]
	N = img.shape[1]
	fm = np.zeros((M,M),dtype=complex)#create inverse fourier matrix of size M
	fm = np.matrix(fm)
	for k in range(M):
		for m in range(M):
			fm[k,m] = np.exp((2.0*1j*np.pi*m*k)/M)
	fn = np.zeros((N,N),dtype=complex)# create inverse fourier matrix of size N
	fn = np.matrix(fn)
	for l in range(N):
		for n in range(N):
			fn[l,n] = np.exp((2.0*1j*np.pi*n*l)/N)
	img = np.matrix(img)
	idftimg = np.matmul(np.matmul(fm,img),fn)/(M*N)# calculate inverse DFT by matrix multiplication of fourier matrices with img array
	return idftimg

##all of the following written by me, ssim from skimage module is used
##function to calculate full inverse
def fullinverse(imgarr,kernelarr):
	pilimg_all = []
	newimg_all = imgarr
	dftkernel = dft2(kernelarr)#dft of kernel
	multfactor = 1/dftkernel #1/dft(kernel)
	for band in range(3): #loop over RGB
		dftimg = np.array(dft2(imgarr[:,:,band]))#dft of image
		dftprod = dftimg*multfactor #element wise multiplication of dft
		idftimg = idft2(dftprod).real.astype('uint8') # computing inverse dft of final image dft
		bf = Image.fromarray(idftimg).convert('L') #image object from numpy array
		newimg_all[:,:,band] = idftimg
		pilimg_all.append(bf)
	finalimg = Image.merge("RGB",(pilimg_all[0],pilimg_all[1],pilimg_all[2]))#merge to RGB
	return finalimg,newimg_all

def truncinverse(imgarr,kernelarr):
	radius = int(raw_input("Please enter truncation radius : ")) #take user input for radius
	modkernel = kernelarr[0:radius,0:radius] # truncate kernel
	pilimg_all = []
	newimg_all = imgarr
	dftkernel = dft2(modkernel)#dft of truncated kernel
	multfactor = 1./dftkernel
	multfactor[np.where(np.abs(multfactor)>=0.71)] = 0.71+0j #thresholding
	for band in range(3):#loop over RGB
		dftimg = np.array(dft2(imgarr[:,:,band]))#dft of image
		dftprod = dftimg
		dftprod[0:radius,0:radius] = dftimg[0:radius,0:radius]*multfactor#element wise multiplication of dft radially limited	
		idftimg = idft2(dftprod).real.astype('uint8') # computing inverse dft of final image dft
		bf = Image.fromarray(idftimg).convert('L') #image object from numpy array
		newimg_all[:,:,band] = idftimg
		pilimg_all.append(bf)
	finalimg = Image.merge("RGB",(pilimg_all[0],pilimg_all[1],pilimg_all[2]))#merge to RGB
	return finalimg,newimg_all

def weinerinverse(imgarr,kernelarr):
	K = float(raw_input("Please enter K value for weiner filtering : "))#take user input for K
	pilimg_all = []
	newimg_all = imgarr
	dftkernel = np.array(dft2(kernelarr))#dft of kernel
	multfactor = np.conj(dftkernel)/(np.abs(dftkernel)**2 + K)  #compute wiener filter
	for band in range(3):#loop over RGB
		dftimg = np.array(dft2(imgarr[:,:,band]))#dft of image
		dftprod = dftimg*multfactor #element wise multiplication of dft
		idftimg = idft2(dftprod).real.astype('uint8') # computing inverse dft of final image dft
		bf = Image.fromarray(idftimg).convert('L') #image object from numpy array
		newimg_all[:,:,band] = idftimg
		pilimg_all.append(bf)
	finalimg = Image.merge("RGB",(pilimg_all[0],pilimg_all[1],pilimg_all[2]))#merge to RGB
	return finalimg,newimg_all

def constrainedls(imgarr,kernelarr):
	gamma = float(raw_input("Please enter gamma value for constrained ls filtering : "))#take user input for gamma
	pilimg_all = []
	newimg_all = imgarr
	p = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]]) ##laplacian matrix
	P = np.pad(p,((0,imgarr.shape[0]-p.shape[0]),(0,imgarr.shape[1]-p.shape[1])),'constant') ##padded laplacian to optimum size
	dftkernel = np.array(dft2(kernelarr))#dft of kernel
	dftp = np.array(dft2(P))#dft of laplacian
	multfactor = np.conj(dftkernel)/(np.abs(dftkernel)**2 + gamma*(np.abs(dftp)**2)) #constrained ls filter
	for band in range(3):#loop over RGB
		dftimg = np.array(dft2(imgarr[:,:,band]))#dft of image
		dftprod = dftimg*multfactor #element wise multiplication of dft
		idftimg = idft2(dftprod).real.astype('uint8') # computing inverse dft of final image dft
		bf = Image.fromarray(idftimg).convert('L') #image object from numpy array
		newimg_all[:,:,band] = idftimg
		pilimg_all.append(bf)
	finalimg = Image.merge("RGB",(pilimg_all[0],pilimg_all[1],pilimg_all[2]))#merge to RGB
	return finalimg,newimg_all

## Command line argument parser to take image file name, kernel file name and output file name
parser = argparse.ArgumentParser()
parser.add_argument('blurred_image',help = "blurred image file name")
parser.add_argument('blur_kernel', help= "blurring kernel file name",default='')
parser.add_argument('outfile', help="output file name",default='out')
parser.add_argument('method', help="choose deblurring method", choices=['fullinv','truncinv','weiner','ls','blind'],default='weiner')
args = parser.parse_args()
## Load image blurred
img = Image.open(args.blurred_image)
imgarr = np.array(img.getdata()).reshape(img.size[1],img.size[0],3)
if(args.method != 'blind'):
	#load ground truth image
	ground = Image.open('GroundTruth1_1_1.jpg')
	groundarr = np.array(ground.getdata()).reshape(ground.size[1],ground.size[0],3)
	#load kernel image
	kernel = Image.open(args.blur_kernel) # open kernel image
	kernelarr = np.array(kernel.getdata()).reshape(kernel.size[1],kernel.size[0]) # get kernel image data
	kernelarr = kernelarr / (np.linalg.norm(kernelarr)**2) * 255. #normalize kernel
	kernelarr2 = np.pad(kernelarr,((0,imgarr.shape[0]-kernelarr.shape[0]),(0,imgarr.shape[1]-kernelarr.shape[1])),'constant') # pad kernel
	ground2 = np.zeros((829,829,3)).astype('uint8')
	for i in range(3):
		ground2[:,:,i] = np.pad(groundarr[:,:,i],((0,imgarr.shape[0]-groundarr.shape[0]),(0,imgarr.shape[1]-groundarr.shape[1])),'constant')# pad ground truth

	ssim_blur = ssim(ground2,imgarr.astype('uint8'),multichannel=True)#ssim of blur image
	psnr_blur = psnr(ground2,imgarr.astype('uint8'))#psnr of blur image

	## if else conditions to call required method and calculate ssim and psnr of restored image
	if(args.method == 'fullinv'):
		bf,newimg = fullinverse(imgarr,kernelarr2)
	elif(args.method == 'truncinv'):
		bf,newimg = truncinverse(imgarr,kernelarr2)
	elif(args.method == 'weiner'):
		bf,newimg = weinerinverse(imgarr,kernelarr2)
	elif(args.method == 'ls'):
		bf,newimg = constrainedls(imgarr,kernelarr2)

	ssim_restored = ssim(ground2,newimg.astype('uint8'),multichannel=True)#ssim of restored image
	psnr_restored = psnr(ground2,newimg.astype('uint8'))#psnr of restored image

	bf.show()#show restored image
	bf.save(args.outfile,'PNG')#save restored image
	print ssim_blur, psnr_blur
	print ssim_restored, psnr_restored ##print ssim and psnr
else: #for blind deblurring
	image = cv2.imread(args.blurred_image)#load image using cv2
	gray = np.array(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)) #change RGB to gray
	kernel = cv2.imread(args.blur_kernel)#load assumed kernel image
	kernel = np.array(cv2.cvtColor(kernel, cv2.COLOR_BGR2GRAY))# RGB to gray
	kernelarr = kernel / (np.linalg.norm(kernel)**2) * 255. #normalize kernel
	kernelarr = np.pad(kernelarr,((0,gray.shape[0]-kernelarr.shape[0]),(0,gray.shape[1]-kernelarr.shape[1])),'constant') # pad kernel
	gamma = float(raw_input("Enter gamma value : "))#get gamma value from user
	p = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]]) ##laplacian matrix
	P = np.pad(p,((0,gray.shape[0]-p.shape[0]),(0,gray.shape[1]-p.shape[1])),'constant') ##padded laplacian to optimum size
	dftkernel = np.array(dft2(kernelarr))#dft of kernel
	dftp = np.array(dft2(P))#dft of laplacian
	multfactor = np.conj(dftkernel)/(np.abs(dftkernel)**2 + gamma*(np.abs(dftp)**2)) #constrained ls filter
	dftimg = np.array(dft2(gray))#dft of image
	dftprod = dftimg*multfactor #element wise multiplication of dft
	idftimg = np.array(idft2(dftprod).real.astype('uint8')) # computing inverse dft of final image dft
	bf = Image.fromarray(idftimg).convert('L') #image object from numpy array
	bf.show()#show image
	bf.save('restored_new.png')#save image
