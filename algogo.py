###### Importation des modules ######

import pandas
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

import warnings
warnings.filterwarnings("ignore")

# ML tools 
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation, Conv2D, MaxPooling2D, Dropout, Conv2D,MaxPooling2D,GlobalAveragePooling2D, BatchNormalization
from keras.optimizers import Adam
from tensorflow.keras import Model
from tensorflow.keras.applications import Xception
import os
from keras import optimizers
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping

###### Utilisation de TPU (si possible) ######

'''
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Device:', tpu.master())
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
except:
    strategy = tf.distribute.get_strategy()
print('Number of replicas:', strategy.num_replicas_in_sync)
'''


###### Importation et changement du dataframe d'apprentissage ######


os.chdir("D:/Tadeg_papiers/Data/ranzcr-clip-catheter-line-classification")
trainset=pandas.read_csv("train.csv")

trainset['path'] = trainset['StudyInstanceUID'].map(lambda x:str('D:/Tadeg_papiers/Data/ranzcr-clip-catheter-line-classification/train/')+x+'.jpg') #ici, on crée une colonne spécifiant le chemin d'accès d'une image de radio pour un patient (à noter qu'il n'y a que des images sous forme de jpg)
trainset = trainset.drop(columns=['StudyInstanceUID','PatientID'])  # plus besoin de cette variable maintenant que le lien des images est dans le dataframe

# print(trainset["path"][1])                              # ici, on vérifie qu'on a bien une image directement accessible depuis le dossier (de type: "chemin d'accès"/"nom d'image".jpg)
# print(trainset.shape)

#### On charge le dataset pour lequel on classe les images ####

testset=pandas.read_csv('sample_submission.csv')
testset['path'] = testset['StudyInstanceUID'].map(lambda x:str('D:/Tadeg_papiers/Data/ranzcr-clip-catheter-line-classification/test/')+x+'.jpg') 
testset = testset.drop(columns=['StudyInstanceUID'])                              
# print(testset)

# On récupère les variables à prédire et on regarde leur distribution

labels = ['ETT - Abnormal', 'ETT - Borderline',
       'ETT - Normal', 'NGT - Abnormal', 'NGT - Borderline',
       'NGT - Incompletely Imaged', 'NGT - Normal', 'CVC - Abnormal',
       'CVC - Borderline', 'CVC - Normal', 'Swan Ganz Catheter Present']

for label in labels:
    print("#"*25)
    print(label)
    print(trainset[label].value_counts(normalize=True) * 100)



####### Split de la base d'apprentissage) #########

X_train, X_valid = train_test_split(trainset, test_size = 0.15, shuffle=True)



####### Transformation des données en tensor #########

train_ds = tf.data.Dataset.from_tensor_slices((X_train.path.values, X_train[labels].values))
valid_ds = tf.data.Dataset.from_tensor_slices((X_valid.path.values, X_valid[labels].values))

for path, label in train_ds.take(5):                             # exemple de chemin d'image avec labels associés (pour la base d'apprentissage, puis de validation)
    print ('Path: {}, Label: {}'.format(path, label))

for path, label in valid_ds.take(5):
    print ('Path: {}, Label: {}'.format(path, label))


####### Data Generator #########


AUTOTUNE = tf.data.experimental.AUTOTUNE

target_size_dim = 300

def process_data_train(image_path, label):
    # load the raw data from the file as a string
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.random_brightness(img, 0.3)
    img = tf.image.random_flip_left_right(img)
    img = tf.image.resize(img, [target_size_dim,target_size_dim])
    return img, label

def process_data_valid(image_path, label):
    # load the raw data from the file as a string
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [target_size_dim,target_size_dim])
    return img, label

# Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
train_ds = train_ds.map(process_data_train, num_parallel_calls=AUTOTUNE)
valid_ds = valid_ds.map(process_data_valid, num_parallel_calls=AUTOTUNE)

'''
for image, label in train_ds.take(1):                   # Affiche une image aléatoire de la base d'apprentissage et les labels associés
    plt.imshow(image.numpy().astype('uint8'))
    plt.show()
    print("Image shape: ", image.numpy().shape)
    print("Label: ", labels[np.argmax(label.numpy())])
'''



def configure_for_performance(ds, batch_size = 16):
    ds = ds.repeat()
    ds = ds.shuffle(buffer_size=256)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds

batch_size = 16

train_ds_batch = configure_for_performance(train_ds)
valid_ds_batch = valid_ds.batch(batch_size*2)

image_batch, label_batch = next(iter(train_ds))



####### Data Augmentation #########


data_augmentation = keras.Sequential(
    [
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.5, interpolation='nearest'),
        tf.keras.layers.experimental.preprocessing.RandomContrast((0.2))
    ]
)




####### Création du modèle (de type EfficientNet) #########


from tensorflow.keras.applications import EfficientNetB3 



def load_pretrained_model(weights_path, drop_connect, target_size_dim, layers_to_unfreeze=5):
    model = EfficientNetB3(
            weights=None, 
            include_top=False, 
            drop_connect_rate=0.4
        )
    
    model.load_weights(weights_path)
    
    model.trainable = True

    return model

def build_my_model(base_model, optimizer, metrics, loss):
    
    inputs = tf.keras.layers.Input(shape=(target_size_dim, target_size_dim, 3))
    x = data_augmentation(inputs)
    outputs_eff = base_model(x)
    global_avg_pooling = GlobalAveragePooling2D()(outputs_eff)
    dense_1= Dense(256)(global_avg_pooling)
    bn_1 = BatchNormalization()(dense_1)
    activation = Activation('relu')(bn_1)
    dropout = Dropout(0.3)(activation)
    dense_2 = Dense(len(labels), activation='sigmoid')(dropout)

    my_model = tf.keras.Model(inputs, dense_2)
    
    my_model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )
    return my_model


############## Téléchargement et chargement du modèle (EfficientNetB3) via https://storage.googleapis.com/keras-applications/efficientnetb3_notop.h5 ############

model_weights_path = 'D:/Tadeg_papiers/Data/ranzcr-clip-catheter-line-classification/efficientnetb3_notop.h5'
print(model_weights_path)

drop_rate = 0.3
base_model = load_pretrained_model(model_weights_path, drop_rate, target_size_dim)

optimizer = tf.keras.optimizers.Adam(lr =0.001)
metrics = tf.keras.metrics.AUC(multi_label=True)

my_model = build_my_model(base_model, optimizer, metrics = [metrics], loss='binary_crossentropy')
my_model.summary()




############ Callbacks ##############

weight_path_save = 'best_model.hdf5'
last_weight_path = 'last_model.hdf5'

checkpoint = ModelCheckpoint(weight_path_save, 
                             monitor= 'val_loss', 
                             verbose=1, 
                             save_best_only=True, 
                             mode= 'min', 
                             save_weights_only = False)
checkpoint_last = ModelCheckpoint(last_weight_path, 
                             monitor= 'val_loss', 
                             verbose=1, 
                             save_best_only=False, 
                             mode= 'min', 
                             save_weights_only = False)


early = EarlyStopping(monitor= 'val_loss', 
                      mode= 'min', 
                      patience=5)

reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=2, verbose=1, mode='auto', min_delta=0.0001, cooldown=5, min_lr=0.00001)
callbacks_list = [checkpoint, checkpoint_last, early, reduceLROnPlat]

 


############ Apprentissage du modèle ##############

epochs = 10

steps_per_epoch = len(X_train) // batch_size
history = my_model.fit(train_ds_batch, 
                        validation_data = valid_ds_batch, 
                        epochs = epochs, 
                        callbacks = callbacks_list,
                        steps_per_epoch = steps_per_epoch
                        )





############ Plot de la perte par epoch (val et train) ##############

def plot_hist(hist):
    plt.figure(figsize=(15,5))
    
    plt2 = plt.gca().twinx()
    plt2.plot(np.arange(local_epochs) ,history.history['loss'],'-o',label='Train Loss',color='#2ca02c')
    plt2.plot(np.arange(local_epochs) ,history.history['val_loss'],'-o',label='Val Loss',color='#d62728')
    plt.legend(loc=2)
    plt.ylabel('Loss',size=14)
    plt.title("Model Accuracy and loss")
    
    plt.savefig('loss.png')
    plt.show()





############ Classification des images de la base de test  ##############


df_test=df_test = pandas.DataFrame(np.array(testset["path"]), columns=['Path'])
print(df_test)

test_ds = tf.data.Dataset.from_tensor_slices((df_test.Path.values))

def process_test(image_path):
    # load the raw data from the file as a string
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [target_size_dim,target_size_dim])
    return img
    
test_ds = test_ds.map(process_test, num_parallel_calls=AUTOTUNE).batch(batch_size*2)

pred_y = my_model.predict(test_ds, workers=4, verbose=1)

df_subs = pandas.DataFrame(pred_y, columns = labels)

df_test['image_id'] = df_test.Path.str.split('/').str[-1].str[:-4]
df_ss['StudyInstanceUID'] = df_test['image_id']
print(df_ss.head())


cols_reordered = ['StudyInstanceUID', 'ETT - Abnormal', 'ETT - Borderline', 'ETT - Normal', 'NGT - Abnormal',
       'NGT - Borderline', 'NGT - Incompletely Imaged', 'NGT - Normal',
       'CVC - Abnormal', 'CVC - Borderline', 'CVC - Normal',
       'Swan Ganz Catheter Present']

df_order = df_ss[cols_reordered]

df_order.to_csv('submission.csv', index=False)
