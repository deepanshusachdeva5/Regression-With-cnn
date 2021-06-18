from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from helper_func import datasets, models
import numpy as np
import argparse
import locale
import os


ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type=str, required=True,
                help="path to the input dataset")
args = vars(ap.parse_args())

print("[INFO] loading house attributes")
inputPath = os.path.sep.join([args['dataset'], 'HousesInfo.txt'])
df = datasets.load_house_attributes(inputPath)


(train, test) = train_test_split(df, test_size=0.25, random_state=42)

maxPrice = train['Price'].max()
trainY = train['Price']/maxPrice
testY = test['Price']/maxPrice


trainX, testX = datasets.process_house_attributes(df, train, test)

model = models.create_mlp(trainX.shape[1], regress=True)
opt = Adam(lr=1e-3, decay=1e-3/200)
model.compile(loss='mean_absolute_percentage_error', optimizer=opt)


print("[INFO] starting training model...")
model.fit(trainX, trainY, validation_data=(
    testX, testY), epochs=200, batch_size=8)

preds = model.predict(testX)

diff = preds.flatten()-testY
percentDiff = (diff/testY)*100
absPercentDiff = np.abs(percentDiff)

mean = np.mean(absPercentDiff)
std = np.std(absPercentDiff)

locale.setlocale(locale.LC_ALL, "en_US.UTF-8")
print("[INFO] avg. house price : {}, std house price : {}".format(locale.currency(
    df['Price'].mean(), grouping=True), locale.currency(df['Price'].std(), grouping=True)))
print("[INFO] mean : {:.2f} , std : {:.2f}".format(mean, std))
