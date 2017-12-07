from  svc_tools import *
from  svc_utils import *
from sklearn import svm
computeFeatures = False
trainClassifier = False
evalClassifier = True

vehicleFeaturesFilename = 'vehicle_features.pkl'
nonVehicleFeaturesFilename = 'nonvehicle_features.pkl'
svcModelFileName = 'cardetector.pkl'

if not os.path.isfile(svcModelFileName):
    print('No trained classifier found, try to train')
    trainClassifier = True
else:
    trainClassifier = False
    print('Trained classifier found')
    print("Load trained classifier")
    svc = loadObject(svcModelFileName)
    if svc == None:
        print("Broken classifier, try to retrain")
        trainClassifier = True
    else:
        print("Success")
        if evalClassifier == False:
            sys.exit(0)
    
if os.path.isfile(vehicleFeaturesFilename) and os.path.isfile(nonVehicleFeaturesFilename):
    computeFeatures = True
    print('Precomputed features found')
    print("Load precomputed features")
    vehicle_features = loadObject(vehicleFeaturesFilename)
    nonvehicle_features = loadObject(nonVehicleFeaturesFilename)
    if vehicle_features==None or len(vehicle_features) == 0 or nonvehicle_features == None or len(nonvehicle_features)==0:
        print("Fail to load precomputed features, remove *.pkl files and rerun train.py")
        sys.exit(1)
else:
    print('No precomputed features found, try to compute')
    vehicles = findFiles('./data/vehicles/**/*.png')
    nonvehicles = findFiles('./data/non-vehicles/**/*.png')

    if len(vehicles) == 0 or len(nonvehicles) == 0:
        print("No train data found, check data exists at: ./data/vehicles and ./data/non-vehicles")
        sys.exit(1)
    
    print("Start processing vehicle features")
    vehicle_features = getFeaturesBatch(filenames=vehicles, printProgress=True)
    saveObject(vehicle_features, vehicleFeaturesFilename)

    print("Start processing nonvehicle features")
    nonvehicle_features = getFeaturesBatch(filenames=nonvehicles, printProgress=True)
    saveObject(nonvehicle_features, nonVehicleFeaturesFilename)

if evalClassifier == True:
    X = np.vstack((vehicle_features, nonvehicle_features)).astype(np.float64)
    X_scaler = StandardScaler().fit(X)
    scaled_X = X_scaler.transform(X)

    y = np.hstack((np.ones(len(vehicle_features)), np.zeros(len(nonvehicle_features))))

    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=42)

    if trainClassifier == True:
        svc = LinearSVC()
        # Check the training time for the SVC
        t=time.time()
        print("Start training classifier")
        svc.fit(X_train, y_train)
        t2 = time.time()
        print(round(t2-t, 2), 'Seconds to train SVC...')
        saveObject(svc, svcModelFileName)

    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    sys.exit(0)

sys.exit(0)
    
    # Check the prediction time for a single sample
    #t=time.time()
    #n_predict = len(X_test)-1
    #print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
    #print('For these',n_predict, 'labels: ', y_test[0:n_predict])
    #t2 = time.time()
    #print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')





