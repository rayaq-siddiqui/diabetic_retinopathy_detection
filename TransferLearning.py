# command to run the code
# python TransferLearning.py --path 'kaggle/' --class_folders '["class0","class1","class2","class3","class4"]' --dim 224 --lr 1e-4 --batch_size 16 --epochs 20 --initial_layers_to_freeze 10 --model InceptionV3 --folds 5 --outdir 'Transfer_Learning_DR/' --mode 'train'

# data science libraries
from unittest import result
from xml.etree.ElementInclude import include
import pandas as pd
import cv2
import numpy as np
np.random.seed(1000)

# sklearn
from sklearn.model_selection import KFold
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import log_loss

# keras and tf
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, GlobalMaxPooling2D, GlobalAveragePooling2D, Convolution2D, MaxPooling2D, ZeroPadding2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, Callback
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import __version__ as keras_version
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers

# resources
import os
import glob
import datetime
import time
import warnings
warnings.filterwarnings("ignore")
import h5py
import argparse
import json
import joblib
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# print validation statement
print("all resources loaded")


# class for transfer learning
class TransferLearning:

    # constructor
    def __init__(self):
        # parsing the input from command line
        parser = argparse.ArgumentParser(description='Process the inputs')
        parser.add_argument('--path',help='image directory')
        parser.add_argument('--class_folders',help='class images folder names')
        parser.add_argument('--dim',type=int,help='Image dimensions to process')
        parser.add_argument('--lr',type=float,help='learning rate',default=1e-4)
        parser.add_argument('--batch_size',type=int,help='batch size')
        parser.add_argument('--epochs',type=int,help='no of epochs to train')
        parser.add_argument('--initial_layers_to_freeze',type=int,help='the initial layers to freeze')
        parser.add_argument('--model',help='Standard Model to load',default='InceptionV3')
        parser.add_argument('--folds',type=int,help='num of cross validation folds',default=5)
        parser.add_argument('--outdir',help='output directory')
        parser.add_argument('--mode',help='train or validation')

        # parsing and understanding the args
        args = parser.parse_args()
        self.path = args.path
        self.class_folders = json.loads(args.class_folders)
        self.dim = int(args.dim)
        self.lr = float(args.lr)
        self.batch_size = int(args.batch_size)
        self.epochs = int(args.epochs)
        self.initial_layers_to_freeze = int(args.initial_layers_to_freeze)
        self.model = args.model
        self.folds = int(args.folds)
        self.outdir = args.outdir
        self.mode = args.mode


    # DEFINING FUNCTIONS
    # getting the image itself
    def get_im_cv2(self, path, dim=224):
        img = cv2.imread(path)
        resized = cv2.resize(img, (dim, dim), cv2.INTER_LINEAR)
        return resized


    # preprocess the image based on the ImageNet pretrained model
    def pre_process(self, img):
        img[:,:,0] = img[:,:,0] - 103.939
        img[:,:,1] = img[:,:,1] - 116.779
        img[:,:,2] = img[:,:,2] - 123.68
        return img


    # function that builds the X,y into numpy format
    def read_data(self, class_folders, path, num_classes, dim, train_val='train'):
        print(train_val)
        train_X, train_y = [], []

        for c in class_folders:
            path_class = path + str(train_val) + '/' + str(c)
            file_list = os.listdir(path_class)

            for f in file_list:
                img = self.get_im_cv2(path_class + '/' + str(f))
                img = self.pre_process(img)
                train_X.append(img)
                train_y.append(int(c.split('class')[1]))

        train_y = to_categorical(np.array(train_y), num_classes)

        return np.array(train_X), train_y


    # we are going to be training the following three models using their pseudocode
    # we are going to train the inceptionV3 architecture here
    def inception_pseudo(self, dim=224, freeze_layers=30, full_freeze='N'):
        model = InceptionV3(weights='imagenet', include_top=False)
        x = model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        out = Dense(5, activation='softmax')(x)
        model_final = Model(model.input, out)

        if full_freeze != 'N':
            for layer in model.layers[0:freeze_layers]:
                layer.trainable = False
        return model_final

    # we are going to train the resnet arhcitecture here
    def resnet_pseudo(self,dim=224,freeze_layers=10,full_freeze='N'):
        model = ResNet50(weights='imagenet',include_top=False)
        x = model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        out = Dense(5,activation='softmax')(x)
        model_final = Model(model.input, out)
        if full_freeze != 'N':
            for layer in model.layers[0:freeze_layers]:
                layer.trainable = False
        return model_final

    # we are going to train the VGG16 architecture here
    def VGG16_pseudo(self,dim=224,freeze_layers=10,full_freeze='N'):
        model = VGG16(weights='imagenet',include_top=False)
        x = model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        out = Dense(5,activation='softmax')(x)
        model_final = Model(model.input, out)
        if full_freeze != 'N':
            for layer in model.layers[0:freeze_layers]:
                layer.trainable = False
        return model_final


    def train_model(self, train_X, train_y, n_fold=5, batch_size=16, epochs=40, dim=224, lr=1e-5, model='ResNet50'):
        model_save_dest = {}
        k = 0
        kf = KFold(n_splits=n_fold, random_state=0, shuffle=True)

        for train_index, test_index in kf.split(train_X):
            k += 1
            X_train, X_test = train_X[train_index], train_X[test_index]
            y_train, y_test = train_y[train_index], train_y[test_index]

            if model == 'Resnet50':
                model_final = self.resnet_pseudo(dim=224,freeze_layers=10,full_freeze='N')

            if model == 'VGG16':
                model_final = self.VGG16_pseudo(dim=224,freeze_layers=10,full_freeze='N')

            if model == 'InceptionV3':
                model_final = self.inception_pseudo(dim=224,freeze_layers=10,full_freeze='N')

            datagen = ImageDataGenerator(
                horizontal_flip = True,
                vertical_flip = True,
                width_shift_range = 0.1,
                height_shift_range = 0.1,
                channel_shift_range = 0,
                zoom_range = 0.2,
                rotation_range = 20
            )

            adam = optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.0)
            model_final.compile(optimizer=adam, loss=["categorical_crossentropy"], metrics=["accuracy"])
            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.50, patience=3, min_lr=0.000001)

            callbacks = [
                EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1),
                CSVLogger('keras-5fold-run-01-v1-epochs_ib.log', separator=',', append=False),
                reduce_lr,
                ModelCheckpoint(
                    'kera1-5fold-run-01-v1-fold-' + str('%02d' % (k + 1)) + '-run-' + str('%02d' % (1 + 1)) + '.check',
                    monitor='val_loss', mode='min',
                    save_best_only=True,
                    verbose=1
                )
            ]

            # note the class weights - they play a part for the imbalanced data
            model_final.fit_generator(
                datagen.flow(X_train, y_train, batch_size=batch_size),
                steps_per_epoch=X_train.shape[0]/batch_size,
                epochs=epochs,
                verbose=1,
			    validation_data=(X_test,y_test),
                callbacks=callbacks,
                class_weight={0:0.012,1:0.12,2:0.058,3:0.36,4:0.43}
            )

            model_name = 'kera1-5fold-run-01-v1-fold-' + str('%02d' % (k + 1)) + '-run-' + str('%02d' % (1 + 1)) + '.check'

            del model_final
            f = h5py.File(model_name, 'r+')
            del f['optimizer_weights']
            f.close()

            model_final = tf.keras.models.load_model(model_name)
            model_name1 = self.outdir + str(model) + '___' + str(k) 
            model_final.save(model_name1)
            model_save_dest[k] = model_name1

        return model_save_dest


    # inference function
    def inference_validation(self, test_X, test_y, model_save_dest, n_class=5, folds=5):
        pred = np.zeros((len(test_X), n_class))

        for k in range(1, folds+1):
            model = tf.keras.load_model(model_save_dest[k])
            pred = pred + model.predict(test_X)
        pred = pred/(1.0*folds)
        pred_class = np.argmax(pred,axis=1)
        act_class = np.argmax(test_y,axis=1)
        accuracy = np.sum([pred_class == act_class])*1.0/len(test_X)
        kappa = cohen_kappa_score(pred_class,act_class,weights='quadratic')
        return pred_class,accuracy,kappa


    # main process
    def main(self):
        start_time = time.time()
        self.num_classes = len(self.class_folders)

        if self.mode == 'train':
            print('Data Processing..')
            file_list, labels = self.read_data(
                self.class_folders,
                self.path,
                self.num_classes,
                self.dim,
                train_val='train'
            )
            print(len(file_list), len(labels))
            print(labels[0], labels[-1])
            self.model_save_dest = self.train_model(
                file_list,
                labels,
                n_fold = self.folds,
                batch_size=self.batch_size,
                epochs = self.epochs,
                dim = self.dim,
                lr = self.lr,
                model = self.model
            )
            joblib.dump(self.model_save_dest, f'{self.outdir}/model_dict.pkl')
            print("Model saved to dest:",self.model_save_dest)
        else:
            model_save_dest = joblib.load(self.model_save_dest)
            print('Models loaded from:', model_save_dest)

            # do inference/validation
            test_files, test_y = self.read_data(
                self.class_folders,
                self.path,
                self.num_classes,
                self.dim,
                train_val='validation'
            )
            test_X = []

            for f in test_files:
                img = self.get_im_cv2(f)
                img = self.pre_process(img)
                test_X.append(img)

            test_X = np.array(test_X)
            test_y = np.array(test_y)
            print(test_X.shape)
            print(test_y.shape)
            pred_class, accuracy, kappa = self.inference_validation(
                test_X,
                test_y,
                model_save_dest,
                n_class=self.num_classes,
                folds=self.folds
            )
            results_df = pd.DataFrame()
            results_df['file_name'] = test_files
            results_df['target'] = test_y
            results_df['prediction'] = pred_class
            results_df.to_csv(f'{self.outdir}/val_results_reg.csv', index=False)
            print("-----------------------------------------------------")
            print("Kappa score:", kappa)
            print("accuracy:", accuracy)
            print("End of training")
            print("-----------------------------------------------------")
            print("Processing Time",time.time() - start_time,' secs')


if __name__ == "__main__":
    obj = TransferLearning()
    obj.main()
