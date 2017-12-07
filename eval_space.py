from  svc_tools import *
from  svc_common_udacity import *
from skimage import exposure
import sys

vehicles = findFiles('./data/vehicles/**/*.png')
nonvehicles = findFiles('./data/non-vehicles/**/*.png')

def getHogChannel(img, conv, channel):
    if conv!=None:
        img = convertColor(img, conv)
        
    f, hogImage = get_hog_features(img[:,:,channel], orient=11, pix_per_cell=16, cell_per_block=2, 
                        vis=True, feature_vec=True)

    hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
    hogImage = hogImage.astype("uint8")
    return hogImage
    
def getChannel(img, conv, channel):
    if conv!=None:
        img = convertColor(img, conv)
    res = np.zeros_like(img)
    res[:,:,0] = img[:,:,channel]
    res[:,:,1] = img[:,:,channel]
    res[:,:,2] = img[:,:,channel]
    res = exposure.rescale_intensity(res, out_range=(0, 255))
    return res
    
def getVehicle(pos):
    return cv2.cvtColor(cv2.imread(vehicles[pos]), cv2.COLOR_BGR2RGB)

def getNonVehicle(pos):
    return cv2.cvtColor(cv2.imread(nonvehicles[pos]), cv2.COLOR_BGR2RGB)

def add(img, pos, name):
    plt.subplot(pos)
    plt.imshow(img)
    plt.axis('off')
    plt.title(name, fontsize=10)

def add2(axarr, img, pos, name):
    axarr[pos[0],pos[1]].imshow(img)
    axarr[pos[0],pos[1]].set_title(name)
    axarr[pos[0],pos[1]].axis('off')
    
def showSpace(conv):
    imgVehicle = getVehicle(100)
    imgNonVehicle = getNonVehicle(100)
    f, axarr = plt.subplots(3, 4)

    add2(axarr,getChannel(imgVehicle, conv, 0), [0,0], 'Vehicle CH-0')
    add2(axarr,getHogChannel(imgVehicle, conv, 0), [0,1], 'Vehicle HOG CH-0')
    add2(axarr,getChannel(imgNonVehicle, conv, 0), [0,2], 'Non vehicle CH-0')
    add2(axarr,getHogChannel(imgNonVehicle, conv, 0), [0,3], 'Non vehicle HOG CH-0')


    add2(axarr,getChannel(imgVehicle, conv, 1), [1,0], 'Vehicle CH-1')
    add2(axarr,getHogChannel(imgVehicle, conv, 1), [1,1], 'Vehicle HOG CH-1')
    add2(axarr,getChannel(imgNonVehicle, conv, 1), [1,2], 'Non vehicle CH-1')
    add2(axarr,getHogChannel(imgNonVehicle, conv, 1), [1,3], 'Non vehicle HOG CH-1')

    add2(axarr,getChannel(imgVehicle, conv, 2), [2,0], 'Vehicle CH-2')
    add2(axarr,getHogChannel(imgVehicle, conv, 2), [2,1], 'Vehicle HOG CH-2')
    add2(axarr,getChannel(imgNonVehicle, conv, 2), [2,2], 'Non vehicle CH-2')
    add2(axarr,getHogChannel(imgNonVehicle, conv, 2), [2,3], 'Non vehicle HOG CH-2')

    

showSpace('RGB2YCrCb')
#fig = plt.figure(figsize = (5,5))
#add(imgVehicle, 121, 'Vehicle')
#add(imgNonVehicle, 122, 'Non vehicle')


#plt.subplots_adjust(wspace = 1, hspace = 0.7)
#subplt1 = fig.add_subplot(2,1,1)
#subplt1.imshow(imgVehicle)
#subplt1.set_title("Vehicle", fontsize=12)
#subplt1.axis('off')

#subplt2 = fig.add_subplot(2,2,1)
#subplt2.imshow(imgNonVehicle)
#subplt2.set_title("Non vehicle", fontsize=12)
#subplt2.axis('off')
#
plt.show()

        


#def showTest(img):

    #showScaled('RGB2YCrCb_HogChannel'+str(channel), hogImage, 3)
    
    #showScaled('RGB2YCrCb_Channel'+str(channel), res, 3)
    
    #conv = 'RGB2YCrCb'
    #showChannel(img,conv,0)
    #showHogChannel(img,conv,0)

    #showChannel(img,conv,1)
    #showHogChannel(img,conv,1)

    #showChannel(img,conv,2)
    #showHogChannel(img,conv,2)

#img = mpimg.imread('./test_images/test1.jpg')
#img = cv2.imread('./test_images/test1.jpg')
#img = img[400:800, :, :]
#img = getVehicle(100)
#showScaled('RGB2YCrCb_orig', img, 3)
#showTest(img)
#showTest(getNonVehicle(0))

#while True:
#    if cv2.waitKey(25)==27:
#        sys.exit(0)