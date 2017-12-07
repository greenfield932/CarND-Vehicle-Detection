from  svc_tools import *
from  svc_common_udacity import *
from skimage import exposure
from scipy.ndimage.measurements import label
def add2(axarr, img, pos, name = None):
    axarr[pos[0],pos[1]].imshow(img)
    if name!=None:
        axarr[pos[0],pos[1]].set_title(name)
    axarr[pos[0],pos[1]].axis('off')
    

def evaluate(filename, svc, X_scaler):
    img = mpimg.imread(filename)
    vehicle_boxes, totalimg = findCars(img, svc, X_scaler)
    totalimg = draw_boxes(totalimg, vehicle_boxes, (255,0,0), thick=3)
    heat = np.zeros_like(img[:,:,0]).astype(np.float)
    heat = add_heat(heat,vehicle_boxes)
    heat = apply_threshold(heat, 1)
    heatmap = np.clip(heat, 0, 255)
    hm = np.zeros_like(img)
    hm[:,:,0] = heat[:]/4*255
    hm[:,:,1] = heat[:]/4*255
    hm[:,:,2] = heat[:]/4*255
    #labels = label(heatmap)
    #img = draw_labeled_bboxes(img, labels)
    
    return totalimg, hm

vehicleFeaturesFilename = 'vehicle_features.pkl'
nonVehicleFeaturesFilename = 'nonvehicle_features.pkl'
svcModelFileName = 'cardetector.pkl'

svc = loadObject(svcModelFileName)
vehicle_features = loadObject(vehicleFeaturesFilename)
nonvehicle_features = loadObject(nonVehicleFeaturesFilename)
X = np.vstack((vehicle_features, nonvehicle_features)).astype(np.float64)
X_scaler = StandardScaler().fit(X)

imageFiles = findFiles('./test_images/*.jpg')
#f, axarr = plt.subplots(2, 2)

#for i in range(0,1):
#    filename = imageFiles[i]
#    totalimg, heatmap = evaluate(filename, svc, X_scaler)
#    add2(axarr, totalimg, [i,0 ])
#    add2(axarr, heatmap, [i, 1])
#plt.subplots_adjust(wspace=0, hspace=0)
#plt.show()

for i in range(0,len(imageFiles)):
    filename = imageFiles[i]
    totalimg, heatmap = evaluate(filename, svc, X_scaler)
    #showScaled('test'+str(i), convertColor(totalimg, 'RGB2BGR'), 0.5)
    #showScaled('heatmap'+str(i), heatmap, 0.5)
    
    res = np.zeros([totalimg.shape[0], totalimg.shape[1]*2, 3]).astype(np.uint8)
    overlay(res, totalimg, 0,0, 1)
    overlay(res, heatmap, totalimg.shape[1],0, 1)
    showScaled('result'+str(i),convertColor(res, 'RGB2BGR'), 0.5)
#waitExit()
#window_img = find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
#plt.imshow(window_img)
#plt.show()