from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from confusion_matrix import confusionMatrix, plotConfusionMatrix, class_accuracy
import math

def openImage(image, loc = None):
    '''
    reads an image with the given and returns a numpy array
    :param image:
    :return: a numpy array of image
    '''
    img = Image.open(image).convert('L')
    arr = np.asarray(img, dtype=np.int32)
    # print(arr.shape)
    return arr

def saveImage(image_array, name, loc = None):
    '''
    saves the image array 'image_array' as image at the given location 'loc' with the given name
    :param image_array:
    :param name:
    :return:
    '''
    im = Image.fromarray(image_array.astype('uint8'))
    im.save(name)

def showImage(image_array, title=None):
    '''
    :param image_array: a numpy array
    :return:
    '''
    im = Image.fromarray(image_array.astype('uint8'))
    im.show(title="abc")

def getSobelKernel():
    '''
    returns two sobel kernels
    :return:
    '''
    filter_x = np.array([[1, 0,-1], [2, 0,-2],[ 1, 0,-1]])
    filter_y = np.array([[1, 2, 1], [0, 0, 0],[-1,-2,-1]])

    # filter_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    # filter_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    return filter_x, filter_y

def convolution(img, filter):
    '''
    performs convolution of the given image with the given filter
    :param img: numpy array of image
    :param filter: numpy array of filter
    :return: numpy array of the result of the convolution
    '''

    filter = np.flip(filter)

    h_filter, w_filter = filter.shape
    assert h_filter%2==1
    assert w_filter%2==1
    cx_filter = int(w_filter/2)
    cy_filter = int(h_filter/2)

    h_img, w_img = img.shape

    img_padded = np.pad(img, pad_width=cx_filter, mode='edge')

    filtered_img = np.zeros((h_img, w_img))
    for j_img in range(h_img):
        for i_img in range(w_img):
            img_slice = img_padded[j_img: j_img+h_filter, i_img:i_img+w_filter]
            value = np.sum(img_slice * filter)
            filtered_img[j_img][i_img] = abs(value)

    return filtered_img


def gaussianPDF(x, mean, stdev):
    '''
    calculates the gaussian probability density
    :param x: sample
    :param mean: mean
    :param stdev: standard deviation
    :return: probability density function
    '''
    var = math.pow(float(stdev), 2)
    lower = math.pow( (2*math.pi * var), 0.5)
    upper = math.exp( -pow( float(x)-float(mean) ,2) / (2*var) )
    return upper/lower


def gaussianKernel2(filter_size, sigma):
    '''
    computes a gaussian filter
    :param filter_size: size of the gaussian filter that is to be returned
    :param sigma: standard deviation of the gaussian filter
    :return: a numpy array which is a gaussian filter of size filter_size*filter_size
    '''
    gaussian_filter = np.zeros((filter_size, filter_size), np.float32)
    m = filter_size // 2
    n = filter_size // 2

    for x in range(-m, m + 1):
        for y in range(-n, n + 1):
            x1 = 2 * np.pi * (sigma ** 2)
            x2 = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
            gaussian_filter[x + m, y + n] = (1 / x1) * x2
    return gaussian_filter

def gaussianBlur(img, kernel_size=5, sigma=2.5):
    '''
    performs gaussian smoothing of the given image array
    :param img: a numpy array of the given image
    :param kernel_size: size of the gaussian kernel
    :param sigma: standard deviation of the gaussian kernel
    :return: a numpy array of smoothed image
    '''
    kernel = gaussianKernel2(kernel_size, sigma)
    img_smooth = convolution(img, kernel)
    return img_smooth

def averageBlur(img, kernel_size=5):
    '''
    performs average smoothing of the given image array
    :param img: a numpy array of the given image
    :param kernel_size: size of the average smoothing kernel
    :return: a numpy array of smoothed image
    '''
    kernel = (1/kernel_size**2) * np.ones((kernel_size, kernel_size))
    img_smooth = convolution(img, kernel)
    return img_smooth

def gradientOfImage(img):
    '''
    computes gradient of the given image array along x-direction and along y-direction
    :param img: a numpy array of the given image
    :return: two numpy arrays - gradient along X and gradient along y
    '''
    ##Sobel operator kernels.
    sobel_x, sobel_y = getSobelKernel()
    img_grad_x = convolution(img, sobel_x)
    img_grad_y = convolution(img, sobel_y)
    return img_grad_x, img_grad_y

def convertGrayToRGB(img):
    '''
    convert grayscale image to RGB image
    :param img: a numpy array of image
    :return: a numpy array of image with 3 different channels: R, G and B
    '''
    stacked_img = np.stack((img,) * 3, axis=-1)
    return stacked_img

def harrisCornerDetector(img, window_size, k, threshold, blur=None):
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
        print("Using {} blur".format(blur))
        img= gaussianBlur(img, kernel_size=3, sigma=5)
        #showImage(img)
    elif blur == 'Average':
        # smoothing image with average filter
        print("Using {} blur".format(blur))
        img = averageBlur(img, kernel_size=7)
        #showImage(img)

    print("Please wait...")
    height,width = img.shape

    I_x, I_y = gradientOfImage(img)
    Ixx = I_x ** 2
    Ixy = I_y * I_x
    Iyy = I_y ** 2

    # create a window filter of the given window_size and perform convolution
    rectWindow = np.ones((window_size, window_size))
    Sxx = convolution(Ixx, rectWindow)
    Sxy = convolution(Ixy, rectWindow)
    Syy = convolution(Iyy, rectWindow)

    # compute determinant and trace
    det = (Sxx * Syy) - (Sxy ** 2)
    trace = Sxx + Syy

    r = det - k * (trace ** 2)

    corners = []
    flats = []
    edges = []
    img_rgb = convertGrayToRGB(original_img)
    img_corners = img_rgb
    img_flats = img_rgb
    img_edges = img_rgb

    X = []
    Y = []

    for j in range(height):
        for i in range(width):
            covariance_mat = np.array([[Sxx[j, i], Sxy[j, i]],
                                       [Sxy[j, i], Syy[j, i]]])
            lambdas, _ = np.linalg.eig(covariance_mat)

            if r[j,i]>threshold:
                # print(r[j,i])
                corners.append([j,i,r[j,i]])
                img_corners[j,i] = [255,0,0]

                X.append(lambdas)
                Y.append('corner')

            elif r[j,i]< -threshold:
                # print(r[j,i])
                edges.append([j,i,r[j,i]])
                img_edges[j,i] = [0,0,255]

                X.append(lambdas)
                Y.append('edge')

            else:
                # print(r[j,i])
                flats.append([j,i,r[j,i]])
                img_flats[j,i] = [0,255,0]

                X.append(lambdas)
                Y.append('flat')

    return X, Y

def calculateMeanAndSD(X):
    '''
    calculates mean and standard deviation from X
    :param X: list or array of data
    :return: returns mean and standard deviation
    '''
    mean = np.mean(X)
    std = np.std(X)
    return mean, std

def fillMeanAndSD(dictx):
    '''
    calculates mean, standard deviation and inserts into the dictionary object
    :param dictx:
    :return: dictionary object after inserting mean and standard deviation
    '''
    for cls, cls_value in dictx.items():
        for feature, feature_value in cls_value.items():
            mean, sd = calculateMeanAndSD(feature_value["data"])
            dictx[cls][feature]['mean'] = mean
            dictx[cls][feature]['stdev'] = sd
    return dictx

def seggregate(X,Y):
    '''
    takes features and labels as input, and returns a special dictionary object
    :param X: features
    :param Y: labels
    :return: a dictionary object
    eg:  input: X = [[5.1,3.5]     Y = [corner,
                     [4.9,3.0]          edge,
                     [...    ]          corner,
                     ...   ..]]         flat]
    returns the features and labels in the form
    {
        label1: {
                    feature1: {
                                data: [5.1, 4.9, .... ],
                                mean: mean value of above data,
                                stdev: stddev of above data,
                    },
                    feature2: {
                                data: [3.5, 3.0, .... ],
                                mean: mean value of above data,
                                stdev: stddev of above data,
                    }
                },
        label2: {
                },
        .
        .
    }
    '''
    num_features = X.shape[1]
    seggregated_data = {}
    for i, data in enumerate(X):
        features = {}
        for j in range(num_features):
            if str(Y[i]) in seggregated_data:
                pass
            else:
                seggregated_data[str(Y[i])] = {}

            if str(j) in seggregated_data[str(Y[i])]:
                temp = np.append(seggregated_data[str(Y[i])][str(j)]["data"], data[j])
                seggregated_data[str(Y[i])][str(j)]["data"] = temp
            else:
                seggregated_data[str(Y[i])][str(j)] = {
                                                        "data" : np.array([ data[j] ]),
                                                        "mean" : None,
                                                        "stdev"  : None
                                                        }
    return fillMeanAndSD(seggregated_data)


def selectMaxProbability(classes,probabilities):
    '''
    selects the class with max class probability
    :param classes: list of classes
    :param probabilities: list of probabilities
    :return: returns class with max class probability and max-probability
    '''
    #print(classes, probabilities)
    assert len(classes)==len(probabilities)
    max_prob = max(probabilities)
    max_prob_index = probabilities.index(max_prob)
    best_class = classes[max_prob_index]
    return best_class, max_prob


def getPredictedClass(Xsample, seggregated_class):
    '''
    calculates class probabilites and makes a prediction for the class of the sample
    :param Xsample: features/attributes of a test sample
    :param seggregated_class: dictionary object
    :return: returns the predicted class and probability of sample falling in that class
    '''
    classes = []
    class_probabilities = []
    for cls, cls_value in seggregated_class.items():
        #print(cls)
        classes.append(cls)
        class_probability = 1
        for i, attribute in enumerate(Xsample):
            mean = cls_value[str(i)]['mean']
            stdev = cls_value[str(i)]['stdev']
            probability = gaussianPDF(attribute, mean, stdev)
            #print("Probability: ", probability, " sample: ", attribute, " mean: ", mean, " stddev: ", stdev)
            class_probability = class_probability * probability

        class_probabilities.append(class_probability)

    best_class, best_probability = selectMaxProbability(classes, class_probabilities)

    return best_class, best_probability


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

def plot_distribution(seggregated_class):
    '''
    plots lambda1 vs lambda 2 for all the classes: corners, edges and flat region
    :param seggregated_class:
    :return:
    '''
    corners_lambda1 = seggregated_class["corner"]["0"]["data"]
    corners_lambda2 = seggregated_class["corner"]["1"]["data"]
    edges_lambda1 = seggregated_class["edge"]["0"]["data"]
    edges_lambda2 = seggregated_class["edge"]["1"]["data"]
    flat_lambda1 = seggregated_class["flat"]["0"]["data"]
    flat_lambda2 = seggregated_class["flat"]["1"]["data"]

    plt.subplot(1,3, 1)
    plt.scatter(corners_lambda1, corners_lambda2)
    plt.xlabel('lambda-1')
    plt.ylabel('lambda-2')
    plt.title('Corners Distribution')

    plt.subplot(1,3, 2)
    plt.scatter(edges_lambda1, edges_lambda2)
    plt.xlabel('lambda-1')
    plt.ylabel('lambda-2')
    plt.title('Edges Distribution')

    plt.subplot(1,3, 3)
    plt.scatter(flat_lambda1, flat_lambda2)
    plt.xlabel('lambda-1')
    plt.ylabel('lambda-2')
    plt.title('Flats Distribution')

    plt.show()

if __name__ == '__main__':
    img_arr = openImage("../data/input_hcd1.jpg")
    X, Y = harrisCornerDetector(img_arr, window_size=3, k=0.06, threshold=12000,
                                                          blur="Gaussian")
    predictions = []
    seggregated_class = seggregate(np.array(X),Y)
    for i in range(len(X)):
        predicted_class, best_probability = getPredictedClass(X[i], seggregated_class)
        predictions.append(predicted_class)

    accuracy = getAccuracy(Y, predictions)

    # plot_distribution(seggregated_class)
    print("=> Overall accuracy", accuracy)
    cm, labels = confusionMatrix(Y, predictions)
    accuracy, _ = class_accuracy(cm, labels)
    print("=> Class labels: ",labels)
    print("=> Class accuracy: ", accuracy)
    print("=> Class error rate: ", 1-accuracy)

    plotConfusionMatrix(cm, labels, plot = False)
    plt.show()
