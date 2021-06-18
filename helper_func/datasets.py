from sklearn.preprocessing import LabelBinarizer, MinMaxScaler
import pandas as pd
import numpy as np
import glob
import cv2
import os


def load_house_attributes(inputPath):
    cols = ['bedrooms', 'bathrooms', 'area', 'zipcode', 'Price']
    df = pd.read_csv(inputPath, sep=' ', header=None, names=cols)

    zipcodes = df['zipcode'].value_counts().keys().tolist()
    counts = df['zipcode'].value_counts().tolist()

    for zipcode, count in zip(zipcodes, counts):

        if count < 25:
            idxs = df[df['zipcode'] == zipcode].index
            df.drop(idxs, inplace=True)

    return df


def process_house_attributes(df, train, test):
    continious = ['bedrooms', 'bathrooms', 'area']

    cs = MinMaxScaler()
    trainContinious = cs.fit_transform(train[continious])
    testContinious = cs.fit_transform(test[continious])

    zipBinarizer = LabelBinarizer().fit(df['zipcode'])
    trainCategorical = zipBinarizer.transform(train['zipcode'])
    testCategorical = zipBinarizer.transform(test['zipcode'])

    trainX = np.hstack([trainContinious, trainCategorical])
    testX = np.hstack([testContinious, testCategorical])

    return (trainX, testX)


def load_house_images(df, inputPath):
    images = []

    for i in df.index.values:
        basePath = os.path.sep.join([inputPath, "{}_*".format(i+1)])
        housePaths = sorted(list(glob.glob(basePath)))

        inputImages = []
        outputImage = np.zeros((64, 64, 3), dtype='uint8')

        for housePath in housePaths:
            image = cv2.imread(housePath)
            image = cv2.resize(image, (32, 32))

            inputImages.append(image)

        outputImage[:32, :32] = inputImages[0]
        outputImage[:32, 32:64] = inputImages[1]
        outputImage[32:64, 32:64] = inputImages[2]
        outputImage[32:64, :32] = inputImages[3]

        images.append(outputImage)

    return np.array(images)
