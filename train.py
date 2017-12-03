from train_utils import *
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.image as mpimg
import numpy as np
import time

computeFeatures = False
trainClassifier = False

if computeFeatures == True:
    vehicles = findFiles('./data/vehicles/**/*.png')
    nonvehicles = findFiles('./data/non-vehicles/**/*.png')

    #vehicles = vehicles[0:100]
    print("Start processing vehicle features")
    vehicle_features = getFeaturesBatch(vehicles, True)
    saveObject(vehicle_features, 'vehicle_hog.pkl')
    
    #nonvehicles = nonvehicles[0:100]
    print("Start processing nonvehicle features")
    nonvehicle_features = getFeaturesBatch(nonvehicles, True)
    saveObject(nonvehicle_features, 'nonvehicle_hog.pkl')
else:
    print("Load precomputed features")
    vehicle_features = loadObject('vehicle_hog.pkl')
    nonvehicle_features = loadObject('nonvehicle_hog.pkl')

X = np.vstack((vehicle_features, nonvehicle_features)).astype(np.float64)
X_scaler = StandardScaler().fit(X)
scaled_X = X_scaler.transform(X)

y = np.hstack((np.ones(len(vehicle_features)), np.zeros(len(nonvehicle_features))))

X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=42)

svc = LinearSVC()

if trainClassifier == True:
    # Check the training time for the SVC
    t=time.time()
    print("Start training classifier")
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')
    saveObject(svc, 'cardetector.pkl')
else:
    print("Load trained classifier")
    svc = loadObject('cardetector.pkl')

# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# Check the prediction time for a single sample
t=time.time()
n_predict = 10
print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
print('For these',n_predict, 'labels: ', y_test[0:n_predict])
t2 = time.time()
print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')





