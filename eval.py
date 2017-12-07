from  svc_tools import *
from  svc_common_udacity import *
from skimage import exposure

img = mpimg.imread('./test_images/test1.jpg')
red = (255,0,0)
green = (0,255,0)
blue = (0,0,255)
yellow = (255,255,0)
white=(255,255,255)
black = (0,0,0)


vehicleFeaturesFilename = 'vehicle_features.pkl'
nonVehicleFeaturesFilename = 'nonvehicle_features.pkl'
svcModelFileName = 'cardetector.pkl'

svc = loadObject(svcModelFileName)
vehicle_features = loadObject(vehicleFeaturesFilename)
nonvehicle_features = loadObject(nonVehicleFeaturesFilename)
X = np.vstack((vehicle_features, nonvehicle_features)).astype(np.float64)
X_scaler = StandardScaler().fit(X)


vehicle_boxes, totalimg = findCars(img, svc, X_scaler)

totalimg = draw_boxes(totalimg, vehicle_boxes, (255,0,0), thick=3)

showAndExit(convertColor(totalimg, 'RGB2BGR'))
#window_img = find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
#plt.imshow(window_img)
#plt.show()