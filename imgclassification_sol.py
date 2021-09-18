#!/usr/bin/env python

##############
#### Your name: Vineeth harish
##############

import numpy as np
import re
import sklearn
from sklearn import svm, metrics
from skimage import io, feature, filters, exposure, color
import ransac_score
import pickle

class ImageClassifier:
    
    def __init__(self):
        self.classifier = None

    def imread_convert(self, f):
        return io.imread(f).astype(np.uint8)

    def load_data_from_folder(self, dir):
        # read all images into an image collection
        ic = io.ImageCollection(dir+"*.bmp", load_func=self.imread_convert)
        
        #create one large array of image data
        data = io.concatenate_images(ic)
        
        #extract labels from image names
        labels = np.array(ic.files)
        for i, f in enumerate(labels):
            m = re.search("_", f)
            labels[i] = f[len(dir):m.start()]
        
        return(data,labels)

    def extract_image_features(self, data):
        # Please do not modify the header above

        # extract feature vector from image data

        ########################
        ########
        feature_data = []

        for img in data:
            grayscale_imgs = color.rgb2gray(img)
            gamma_adjusted = exposure.adjust_gamma(grayscale_imgs)
            log_adjusted = exposure.adjust_log(gamma_adjusted) 
            sigmoid_adjusted = exposure.adjust_sigmoid(log_adjusted)
            gaussian_blur = filters.gaussian(sigmoid_adjusted, sigma=0.9) 
            sobel_filter = filters.sobel(gaussian_blur)

            hog_imgs = feature.hog(sobel_filter,
                pixels_per_cell=(24,24),
                cells_per_block=(6,6),
                orientations=10,
                block_norm='L2-Hys'
            )
            feature_data.append(hog_imgs)
        #feature_data = np.array([StandardScaler(img) for img in hog_imgs])
        
        ########################

        # Please do not modify the return type below
        return(feature_data)

    def train_classifier(self, train_data, train_labels):
        # Please do not modify the header above
        
        # train model and save the trained model to self.classifier
        
        ########################
        ######## 
        classifier = svm.SVC(random_state= 1, max_iter = 2000, tol= 1e-7)
        classifier.fit(train_data, train_labels)
        self.classifier = classifier
        ########################

    def predict_labels(self, data):
        # Please do not modify the header

        # predict labels of test data using trained model in self.classifier
        # the code below expects output to be stored in predicted_labels
        
        ########################
        ########
        predicted_labels = self.classifier.predict(data)

        ########################

        # Please do not modify the return type below
        return predicted_labels

    def line_fitting(self, data):
        # Please do not modify the header

        # fit a line the to arena wall using RANSAC
        # return two lists containing slopes and y intercepts of the line

        ########################
        slope = []
        intercept = []
        canny_edges = []
        for img in data:
            gray_scale_imgs = color.rgb2gray(img)
            canny_edges.append(np.argwhere(feature.canny(gray_scale_imgs, sigma = 4) == True))
        for edge in canny_edges:
            ransac = sklearn.linear_model.RANSACRegressor(residual_threshold=0.25)
            x_init = np.expand_dims(edge[:,1], axis = 1)
            y_init = edge[:,0]
            ransac.fit(x_init, y_init)
            slope.append(ransac.estimator_.coef_)
            intercept.append(ransac.estimator_.intercept_)
        ########################

        # Please do not modify the return type below
        return slope, intercept
        
def main():

    img_clf = ImageClassifier()

    # load images
    (train_raw, train_labels) = img_clf.load_data_from_folder('./train/')
    (test_raw, test_labels) = img_clf.load_data_from_folder('./test/')
    (wall_raw, _) = img_clf.load_data_from_folder('./wall/')
    
    # convert images into features
    train_data = img_clf.extract_image_features(train_raw)
    test_data = img_clf.extract_image_features(test_raw)
    
    # train model and test on training data
    breakpoint()
    img_clf.train_classifier(train_data, train_labels)
    predicted_labels = img_clf.predict_labels(train_data)
    print("\nTraining results")
    print("=============================")
    print("Confusion Matrix:\n",metrics.confusion_matrix(train_labels, predicted_labels))
    print("Accuracy: ", metrics.accuracy_score(train_labels, predicted_labels))
    print("F1 score: ", metrics.f1_score(train_labels, predicted_labels, average='micro'))
    
    # test model
    pickle.dump(img_clf, open("classifier_pickle", 'wb'))
    predicted_labels = img_clf.predict_labels(test_data)
    print("\nTest results")
    print("=============================")
    print("Confusion Matrix:\n",metrics.confusion_matrix(test_labels, predicted_labels))
    print("Accuracy: ", metrics.accuracy_score(test_labels, predicted_labels))
    print("F1 score: ", metrics.f1_score(test_labels, predicted_labels, average='micro'))

    # ransac
    print("\nRANSAC results")
    print("=============================")
    s, i = img_clf.line_fitting(wall_raw)
    print(f"Line Fitting Score: {ransac_score.score(s,i)}/10")

if __name__ == "__main__":
    main()
