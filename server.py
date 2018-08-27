from __future__ import division
from __future__ import print_function
import numpy as np 
import pandas as pd
import tensorflow as tf
from pubnub.enums import PNStatusCategory
from pubnub.pnconfiguration import PNConfiguration
from pubnub.pubnub import PubNub, SubscribeListener
pnconfig = PNConfiguration()
# for submition only TOBE stored in credential manager
pnconfig.publish_key = "pub-c-ccdc105a-4aa2-4979-b059-c7f839b7dd56"
pnconfig.subscribe_key = "sub-c-f2d06eca-c630-11e7-8d13-12daa3930087"

pubnub = PubNub(pnconfig)

my_listener = SubscribeListener()
pubnub.add_listener(my_listener)


# to be run on start of this server
roadsTrain = pd.read_csv("./data/400carsdatanew.csv")
roadsTrainTest = pd.read_csv("./data/roads.test.csv")
# print(roadsTrain.head(20))
FEATURE_KEYS = ['Day', 'Time', 'RoadNo', 'Cars']
from sklearn.model_selection import train_test_split

roadsTrain.iloc[:,1:4] = roadsTrain.iloc[:,1:4].astype(np.float32)
X_train, X_test, y_train, y_test = train_test_split(roadsTrain.iloc[:,0:4], roadsTrain["Class"], test_size=0.25, random_state=0)

feature_columns = [
    tf.feature_column.numeric_column(key, shape=1) for key in FEATURE_KEYS]

def input_fn(df,labels):
    feature_cols = {k:tf.constant(df[k].values) for k in FEATURE_KEYS}
    label = tf.constant(labels.values)
    return feature_cols,label


classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,hidden_units=[10,20,10],n_classes = 3)
classifier.fit(input_fn=lambda: input_fn(X_train,y_train),steps = 1000)
ev = classifier.evaluate(input_fn=lambda: input_fn(X_test,y_test),steps=1)
print(ev)
def input_predict(df):
    feature_cols = {k:tf.constant(df[k].values,shape = [df[k].size,1]) for k in FEATURE_KEYS}
    print(feature_cols)
    return feature_cols

new_dict = {0: 'LOW', 1: 'MEDIUM', 2:'HIGH'}
new_dictColor = {0: '\033[32m', 1: '\033[33m', 2:'\033[31m'}
pred = classifier.predict_classes(input_fn=lambda: input_predict(roadsTrainTest))
for i in pred:
    print(new_dict[i])

def getUserInput(text):
    data = [int(x) for x in text.split(",")]
    feature_cols1 = {k:tf.constant(data[i],shape = [1,1]) for i, k in enumerate(FEATURE_KEYS)}
    return feature_cols1

def sendMsg(txt):
    try:
        pubnub.publish().channel('Channel-5a7bgoadr').message({'trafficlight': txt }).sync()
    except:
        print('ERR::::X:S::X:S:SX:XS:X:')

pubnub.subscribe().channels('Channel-5a7bgoadr').execute()
print(':Waiting for device Data:')
while(True):
    # Getting message from traffic light PUB_SUB channel 
    result = my_listener.wait_for_message_on('Channel-5a7bgoadr')
    # print(result.message)
    if 'isActiveChannel' in result.message.keys() and result.message['isActiveChannel'] :
        pred1 = classifier.predict_classes(input_fn=lambda: getUserInput(result.message['cars']))
        print('Day,Time,RoadNo,Cars')
        for j in pred1:
            print(new_dictColor[j] + new_dict[j]+ '\033[0m')
            sendMsg(new_dict[j])


