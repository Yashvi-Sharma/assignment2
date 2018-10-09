import numpy as np
import argparse
from PIL import Image , ImageTk, ImageCms
from scipy import ndimage, signal
from skimage.measure import structural_similarity as ssim
import math
from skimage import restoration
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
def fullinverse(imgarr,kernelarr,optsize0,optsize1):
	# idftimg_all = []
	# for band in range(3):
	imgarr = np.pad(imgarr,((0,optsize0-imgarr.shape[0]),(0,optsize1-imgarr.shape[1])),'constant')#pad image to optimum size
	dftimg = np.array(dft2(imgarr))#dft of image
	dftkernel = dft2(kernelarr)#dft of kernel
	multfactor = 1/dftkernel #1/dft(kernel)
	dftprod = dftimg*multfactor #element wise multiplication of dft
	idftimg = idft2(dftprod)*2000 # computing inverse dft of final image dft and scaling intensities by 2000 because they are very low
	bf = Image.fromarray(idftimg.real.astype('uint8'),'L') #image object from numpy array
	return bf,np.array(idftimg.real.astype('uint8'))

def truncinverse(imgarr,kernelarr,optsize0,optsize1):
	radius = int(raw_input("Please enter truncation radius : ")) #take user input for radius
	# idftimg_all = []
	# for band in range(3):
	imgarr = np.pad(imgarr,((0,optsize0-imgarr.shape[0]),(0,optsize1-imgarr.shape[1])),'constant')#pad image to optimum size
	modkernel = kernelarr[0:radius,0:radius] # truncate kernel
	dftimg = np.array(dft2(imgarr))#dft of image
	dftkernel = dft2(modkernel)#dft of truncated kernel
	multfactor = np.conj(dftkernel)/(np.abs(dftkernel)**2) #compute wiener filter
	dftprod = dftimg
	dftprod[0:radius,0:radius] = dftimg[0:radius,0:radius]*multfactor#element wise multiplication of dft radially limited
	idftimg = idft2(dftprod) # computing inverse dft of final image dft 
	bf = Image.fromarray(idftimg.real.astype('uint8'),'L')#image object from numpy array
	return bf,np.array(idftimg.real.astype('uint8'))	

def weinerinverse(imgarr,kernelarr,optsize0,optsize1):
	K = float(raw_input("Please enter K value for weiner filtering : "))#take user input for K
	# idftimg_all = []
	# for band in range(3):
	imgarr = np.pad(imgarr,((0,optsize0-imgarr.shape[0]),(0,optsize1-imgarr.shape[1])),'constant')#pad image to optimum size
	dftimg = np.array(dft2(imgarr))#dft of image
	dftkernel = np.array(dft2(kernelarr))#dft of kernel
	multfactor = np.conj(dftkernel)/(np.abs(dftkernel)**2 + K)
	dftprod = dftimg*multfactor#element wise multiplication of dft
	idftimg = idft2(dftprod)*2000 # computing inverse dft of final image dft and scaling intensities by 2000 because they are very low
	#idftimg = restoration.wiener(imgarr, kernelarr, 100)*1000 ##checked using skimage wiener filter, gives same result of too low intensities
	bf = Image.fromarray(idftimg.real.astype('uint8'),'L')#image object from numpy array
	return bf,np.array(idftimg.real.astype('uint8'))

def constrainedls(imgarr,kernelarr,optsize0,optsize1):
	gamma = float(raw_input("Please enter gamma value for constrained ls filtering : "))#take user input for gamma
	p = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]]) ##laplacian matrix
	P = np.pad(p,((0,optsize0-p.shape[0]),(0,optsize1-p.shape[1])),'constant') ##padded laplacian to optimum size
	imgarr = np.pad(imgarr,((0,optsize0-imgarr.shape[0]),(0,optsize1-imgarr.shape[1])),'constant')#pad image to optimum size
	dftimg = np.array(dft2(imgarr))#dft of image
	dftkernel = np.array(dft2(kernelarr))#dft of kernel
	dftp = np.array(dft2(P))#dft of laplacian
	multfactor = np.conj(dftkernel)/(np.abs(dftkernel)**2 + gamma*(np.abs(dftp)**2)) #constrained ls filter
	dftprod = dftimg*multfactor#element wise multiplication of dft
	idftimg = idft2(dftprod)*2000 # computing inverse dft of final image dft and scaling intensities by 2000 because they are very low
	bf = Image.fromarray(idftimg.real.astype('uint8'),'L')#image object from numpy array
	return bf,np.array(idftimg.real.astype('uint8'))

## Command line argument parser to take image file name, kernel file name and output file name
parser = argparse.ArgumentParser()
parser.add_argument('blurred_image',help = "blurred image file name")
parser.add_argument('blur_kernel', help= "blurring kernel file name",default='')
parser.add_argument('outfile', help="output file name")
parser.add_argument('method', help="choose deblurring method", choices=['fullinv','truncinv','weiner','ls'])
args = parser.parse_args()
## Load image blurred
img = Image.open(args.blurred_image)
#load ground truth image
ground = Image.open('GroundTruth1_1_1.jpg').convert('HSV')
ground2 = list(ground.getdata(band=2))
ground2 = np.array(ground2).reshape(ground.size[1],ground.size[0]).astype('uint8')
img = img.convert('HSV') ##convert to HSV
#img.show()
data = list(img.getdata(band=2)) #get V data
imgarr = np.array(data).reshape(img.size[1],img.size[0]).astype('uint8')#convert image object to array
#Image.fromarray(ground2).show()
ssim0 = ssim(ground2,imgarr)#ssim of blurred image
psnr0 = psnr(ground2,imgarr)#psnr of blurred image
if(args.blur_kernel!=''):
	kernel = Image.open(args.blur_kernel) # open kernel image
	kernel_data = list(kernel.getdata(band=0)) # get kernel image data
	kernelarr = np.array(kernel_data).reshape(kernel.size[1],kernel.size[0]) #reshape to numpy array
	## compute optimum size for padding for DFT calculation
	optsize1 = img.size[0]+kernel.size[0]+2 
	optsize0 = img.size[1]+kernel.size[1]+2
	kernelarr = np.pad(kernelarr,((0,optsize0-kernelarr.shape[0]),(0,optsize1-kernelarr.shape[1])),'constant') # pad kernel
	ground2 = np.pad(ground2,((0,optsize0-ground2.shape[0]),(0,optsize1-ground2.shape[1])),'constant')# pad ground truth
	## if else conditions to call required method and calculate ssim and psnr of restored image
	if(args.method == 'fullinv'):
		bf,newimg = fullinverse(imgarr,kernelarr,optsize0,optsize1)
		ssim1 = ssim(ground2,newimg)
		psnr1 = psnr(ground2,newimg)
	elif(args.method == 'truncinv'):
		bf,newimg = truncinverse(imgarr,kernelarr,optsize0,optsize1)
		ssim1 = ssim(ground2,newimg)
		psnr1 = psnr(ground2,newimg)
	elif(args.method == 'weiner'):
		bf,newimg = weinerinverse(imgarr,kernelarr,optsize0,optsize1)
		ssim1 = ssim(ground2,newimg)
		psnr1 = psnr(ground2,newimg)
	elif(args.method == 'ls'):
		bf,newimg = constrainedls(imgarr,kernelarr,optsize0,optsize1)
		ssim1 = ssim(ground2,newimg)
		psnr1 = psnr(ground2,newimg)
	bf.show()#show restored image
	bf.save(args.outfile,'PNG')#save restored image
	print ssim1, psnr1, ssim0, psnr0##print ssimand psnr
#else:
	##estimate kernel apply weiner







