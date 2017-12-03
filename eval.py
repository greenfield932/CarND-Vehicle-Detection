from train_utils import *
from sliding_windows_detector import *
import matplotlib.image as mpimg

img = mpimg.imread('./test_images/test1.jpg')
red = (255,0,0)
green = (0,255,0)
blue = (0,0,255)
yellow = (255,255,0)
white=(255,255,255)
black = (0,0,0)

svc = loadObject('cardetector.pkl')
vehicle_features = loadObject('vehicle_hog.pkl')
nonvehicle_features = loadObject('nonvehicle_hog.pkl')
X = np.vstack((vehicle_features, nonvehicle_features)).astype(np.float64)
X_scaler = StandardScaler().fit(X)

w = img.shape[1]
h = img.shape[0]
sizes =   [25, 80, 120, 200]
ystarts = [0,   20,  20,  0 ]
ystops =  [60, 100,  200, 350]
#sizes =   [50]
#ystarts = [0]

y0 = h*0.55
window_img = img.copy()
colors = [red, green, blue, white, yellow]
vehicle_boxes = []
t = time.time()
for i in range(0, len(sizes)):

    windows = slide_window(img, x_start_stop=(w*0.25, w), y_start_stop=(y0+ystarts[i], y0+ystarts[i]+ystops[i]), 
                    xy_window=(sizes[i], sizes[i]), xy_overlap=(0.5, 0.5))
    
    for box in windows:
        subimg  = cv2.resize(img[box[0][1]:box[1][1], box[0][0]:box[1][0]], (64, 64))
        features = getFeatures(subimg)
        features = X_scaler.transform(np.array(features).reshape(1, -1))

        pred = svc.predict(features)
        if pred == 1:
            vehicle_boxes.append(box)
t2 = time.time()
print(round(t2-t, 5), 'Seconds to predict with SVC')


    #window_img = draw_boxes(window_img, windows, colors[i], thick=3)
window_img = draw_boxes(window_img, vehicle_boxes, (255,0,0), thick=3)

plt.imshow(window_img)
plt.show()