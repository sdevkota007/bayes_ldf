import os
import numpy as np
import random
import matplotlib.pyplot as plt
from confusion_matrix import confusionMatrix, plotConfusionMatrix, class_accuracy
from classification_with_hcd import *


dataset_path = "../data/DigitDataset/"
num_train_examples = 50
num_test_examples = 50
num_examples = num_train_examples + num_train_examples
num_labels = 10
num_features = 20


def fetch_files(path):
    '''
    fetches all of the files from the given path
    :return: a dictionary object in the form
            {
             'data' : {
                        '0': {
                                'images': [ names of images inside folder 0 ]
                             }
                        '1': {
                                'images': [ names of images inside folder 1 ]
                             }
            }
    '''
    labels = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

    dataset_info = {"data": {} }
    for label in labels:
        images = os.listdir(os.path.join(path, label))[:num_examples]

        dataset_info["data"][label] = {"images": [ os.path.join(path, label, image) for image in images]}

    return dataset_info



def split_train_test(data):
    '''
    splits the given data into training set and test set
    :param data: a dictionary object
    :return: training images, training labels, test images, test labels
    '''
    X_images_test = []
    Y_test = []
    X_images_train = []
    Y_train = []

    for label, value in data['data'].items():
        for i in range(num_test_examples):
            image = value['images'].pop(i)
            X_images_test.append(image)
            Y_test.append(label)

    for label, value in data['data'].items():
        for i in range(num_train_examples):
            image = value['images'][i]
            X_images_train.append(image)
            Y_train.append(label)

    return X_images_train, Y_train, X_images_test, Y_test



def harris_feature_vector(img, window_size, k, threshold, blur=None):
    '''
    Performs harris corner detection on the given image
    :param img: Numpy array of input image
    :param window_size: Size of the gaussian window
    :param k: k is the sensitivity factor to separate corners from edges, typically a value between 0.4 to 0.6
    :param threshold: the threshold above which a corner is counted
    :param blur: if set to "Gaussian', a gaussian smoothing of the image is performed before detecting corners
                else, if it is set to 'Average', an average smoothing of the image is performed before detecting corners
    :return:
    '''

    original_img = img

    if blur == 'Gaussian':
        img = gaussianBlur(img, kernel_size=3, sigma=5)
        # showImage(img)
    elif blur == 'Average':
        # smoothing image with average filter
        img = averageBlur(img, kernel_size=7)
        # showImage(img)

    height, width = img.shape

    I_x, I_y = gradientOfImage(img)
    Ixx = I_x ** 2
    Ixy = I_y * I_x
    Iyy = I_y ** 2

    # create a window filter of the given window_size and perform convolution
    rectWindow = np.ones((window_size, window_size))
    Sxx = convolution(Ixx, rectWindow)
    Sxy = convolution(Ixy, rectWindow)
    Syy = convolution(Iyy, rectWindow)

    # showImage(Sxx)
    # showImage(Sxy)
    # showImage(Syy)

    # compute determinant and trace
    det = (Sxx * Syy) - (Sxy ** 2)
    trace = Sxx + Syy

    r = det - k * (trace ** 2)
    flattened_R = np.ndarray.flatten(r)
    sorted_R = np.sort(flattened_R)
    num_features_half = int(num_features/2)
    R_score = np.concatenate((sorted_R[:num_features_half], sorted_R[-num_features_half:]))
    return R_score


def compute_feature_vector(images):
    '''
    computes R_scores of given images
    :param images: list of images_names
    :return:
    '''
    R_scores = []
    for image in images:
        img_arr = openImage(image)
        R_score = harris_feature_vector(img_arr, window_size=3, k=0.06,
                                        threshold=10000,
                                        blur="Gaussian"
                                        )
        R_scores.append(np.array(R_score).reshape(1,num_features))
    return np.array(R_scores).reshape(-1,num_features)

def compute_h_b(data):
    '''
    computes h-vector and b-value from class mean and overall covariance which are present inside data object
    :param data: dictionary object
    :return: dictionary object after inserting h vector and b value
    '''
    cov_mat_inv = np.linalg.inv(data['overall_cov_mat'])
    for label, value in data["data"].items():
        class_mean = value['class_mean'].reshape(num_features,1)
        h = np.matmul(cov_mat_inv, class_mean)
        b = -(1/2) * np.matmul(np.transpose(class_mean), h)

        data['data'][label]['h'] = h.reshape(1,num_features)
        data['data'][label]['b'] = b[0][0]

    return data


def bayes_ldf_classifier(dataset_info):
    '''
    Computes parameter of Bayes LDF: class mean, class covariance, overall mean and overall covariance of the given
    dataset present inside the dictionary object 'dataset_info'
    :param dataset_info: dictionary object
    :return: dictionary object after inserting parameter of Bayes LDF
    '''
    array_mean_vector = []
    array_R_scores = []
    for label, value in dataset_info["data"].items():
        R_scores = compute_feature_vector(value["images"])

        dataset_info["data"][label]["R_scores"] = R_scores
        array_R_scores.append(R_scores)

        mean_vector = np.mean(R_scores, axis = 0).reshape(-1, num_features)
        dataset_info["data"][label]["class_mean"] = mean_vector

        array_mean_vector.append(mean_vector)

    overall_mean_vector = np.mean(np.array(array_mean_vector), axis=0)
    dataset_info["overall_mean"] = overall_mean_vector


    array_R_scores = np.array(array_R_scores).reshape(-1,20)

    sum_cov_mat = np.zeros((num_features, num_features))
    for label, value in dataset_info["data"].items():
        sum_class_cov_mat = np.zeros((num_features, num_features))
        for R_score in value['R_scores']:
            R_score = R_score.reshape(1,num_features)
            # Calculating class_covariance
            diff = (R_score - value['class_mean'])
            cov_mat = np.matmul(np.transpose(diff), diff)
            sum_class_cov_mat = sum_class_cov_mat + cov_mat

            # Calculating overall_covariance
            diff = (R_score - overall_mean_vector)
            cov_mat = np.matmul(np.transpose(diff), diff)
            sum_cov_mat = sum_cov_mat + cov_mat

        class_cov_mat = (1/ (num_train_examples)) * sum_class_cov_mat
        dataset_info['data'][label]['class_cov_mat'] = class_cov_mat

    overall_cov_mat = (1/ (num_train_examples * num_labels)) * sum_cov_mat
    dataset_info["overall_cov_mat"] = overall_cov_mat

    # a = array_R_scores - overall_mean_vector
    # a_transpose = np.ndarray.transpose(a)
    # covariance_mat =  (1/ (num_train_examples * num_labels)) * np.matmul(a_transpose, a)
    # dataset_info["covariance_mat"] = covariance_mat
    # print( covariance_mat, covariance_mat.shape)

    dataset_info = compute_h_b(dataset_info)

    return dataset_info


def selectMaxProbability(classes,probabilities):
    '''
    selects the class with max class probability
    :param classes: list of classes
    :param probabilities: list of probabilities
    :return: returns class with max class probability and max-probability
    '''
    # print(classes, probabilities)
    assert len(classes)==len(probabilities)
    max_prob = max(probabilities)
    max_prob_index = probabilities.index(max_prob)
    best_class = classes[max_prob_index]
    return best_class, max_prob


def getAccuracy(true_labels, predictions):
    '''
    calculates accuracy
    :param true_labels:
    :param predictions:
    :return: returns accuracy
    '''
    result = list(map(lambda x,y: (1 if x==y else 0), predictions, true_labels))
    accuracy = sum(result)/(len(result))
    return accuracy


def make_prediction(test_images, train_data):
    '''
    takes list of input images, and returns predictions for test images
    :param test_images:
    :param train_data:
    :return:
    '''
    R_scores = compute_feature_vector(test_images)
    # print(R_scores, R_scores.shape)

    predictions = []
    for i, image in enumerate(test_images):
        g_xs = []
        labels = []
        for label, value in train_data["data"].items():
            h = value['h'].reshape(num_features,1)
            b = value['b']
            x = R_scores[i].reshape(1, num_features)
            g_x = np.matmul(x,h) + b
            g_xs.append(g_x[0][0])
            labels.append(label)
        best_class, max_prob = selectMaxProbability(labels, g_xs)
        predictions.append(best_class)
    return predictions



def main():
    dataset_info = fetch_files(dataset_path)
    X_images_train, Y_train, X_images_test, Y_test = split_train_test(dataset_info)

    train_data = bayes_ldf_classifier(dataset_info)

    # # ****************************************************************************
    # # for test accuracy
    predictions_test = make_prediction(X_images_test, train_data)
    accuracy = getAccuracy(Y_test, predictions_test)
    cm, labels = confusionMatrix(Y_test, predictions_test)
    class_accuracies, _ = class_accuracy(cm, labels)
    print("************For test set*******************")
    print("=> Overall accuracy", accuracy)
    print("=> Class labels: ",labels)
    print("=> Class accuracy: ", class_accuracies)
    print("=> Class error rate: ", 1-class_accuracies)

    plotConfusionMatrix(cm, labels, plot = False)
    plt.show()

    # # ****************************************************************************
    # # for training accuracy
    predictions_train = make_prediction(X_images_train, train_data)
    accuracy = getAccuracy(Y_train, predictions_train)
    cm, labels = confusionMatrix(Y_train, predictions_train)
    class_accuracies, _ = class_accuracy(cm, labels)
    print("************For training set*******************")
    print("=> Overall accuracy", accuracy)
    print("=> Class labels: ",labels)
    print("=> Class accuracy: ", class_accuracies)
    print("=> Class error rate: ", 1-class_accuracies)

    plotConfusionMatrix(cm, labels, plot=False)
    plt.show()

if __name__ == '__main__':
    main()