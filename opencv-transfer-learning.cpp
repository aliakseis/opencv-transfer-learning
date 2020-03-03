// opencv-transfer-learning.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

/*

imho, dnn's rule for this kind of task, nowadays.

we could try to use transfer learning ,

that is: use an existing, pretrained model, and try to teach it some new tricks !

we can just "pipe" our images through the network, stop it at some layer (before it would do the final classification), 
grab the output neurons from there, and feed our own ml classifier with this data (instead of using the "raw" images) , like this:

(colour) image   --> DNN --> 1000 numbers  --> our own classifier (ANN_MLP for today)
since opencv's dnn module already supports various classification models, let's try with squeezenet (which is also small, and quite fast !)
https://arxiv.org/abs/1602.07360

it was trained on millions of images (imagenet), among them cats & dogs. so, it has "seen the world", already. ;)

there are 67 layers (!), here's how the last 10 look like: (i=input,o=output)

fire9/squeeze1x1                       Convolution   i[1, 512, 14, 14]  o[1, 64, 14, 14]
fire9/relu_squeeze1x1                  ReLU          i[1, 64, 14, 14]  o[1, 64, 14, 14]
fire9/expand1x1                        Convolution   i[1, 64, 14, 14]  o[1, 256, 14, 14]
fire9/relu_expand1x1                   ReLU          i[1, 256, 14, 14]  o[1, 256, 14, 14]
fire9/expand3x3                        Convolution   i[1, 64, 14, 14]  o[1, 256, 14, 14]
fire9/relu_expand3x3                   ReLU          i[1, 256, 14, 14]  o[1, 256, 14, 14]
fire9/concat                           Concat        i[1, 256, 14, 14]  i[1, 256, 14, 14]  o[1, 512, 14, 14]
drop9                                  Dropout       i[1, 512, 14, 14]  o[1, 512, 14, 14]
conv10                                 Convolution   i[1, 512, 14, 14]  o[1, 1000, 14, 14]
relu_conv10                            ReLU          i[1, 1000, 14, 14]  o[1, 1000, 14, 14]
pool10                                 Pooling       i[1, 1000, 14, 14]  o[1, 1000, 1, 1]
prob                                   Softmax       i[1, 1000, 1, 1]  o[1, 1000, 1, 1]
so, pool10 looks like a good place to tap it !

(1000 features are a good number, if we have ~1000 images in our dataset)

you'll need to download the caffemodel and the prototxt , then we can start playing with your cats vs dogs dataset
https://raw.githubusercontent.com/DeepScale/SqueezeNet/b5c3f1a23713c8b3fd7b801d229f6b04c64374a5/SqueezeNet_v1.1/squeezenet_v1.1.caffemodel
https://github.com/opencv/opencv_extra/blob/master/testdata/dnn/squeezenet_v1.1.prototxt
https://github.com/yoggasek/Train_Data

https://answers.opencv.org/question/191359/ml-svm-k-nn-image-recognition-examples-in-c/

*/

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
    vector<String> fn;
    glob("c:/data/cat-dog/*.jpg", fn, true);
    // glob() will conveniently sort names lexically, so the cats come first!
    // so we have 700 cats, 699 dogs, and split it into:
    // 100 test cats
    // 600 train cats
    // 100 test dogs
    // 599 train dogs

    std::string modelTxt = "c:/data/mdl/squeezenet/deploy.prototxt";
    std::string modelBin = "c:/data/mdl/squeezenet/squeezenet_v1.1.caffemodel";
    dnn::Net net = dnn::readNetFromCaffe(modelTxt, modelBin);
    cv::Size inputImgSize = cv::Size(227, 227); // model was trained with this size

    Mat_<int> layers(4, 1);
    layers << 1000, 400, 100, 2; // the sqeezenet pool10 layer has 1000 neurons

    Ptr<ml::ANN_MLP> nn = ml::ANN_MLP::create();
    nn->setLayerSizes(layers);
    nn->setTrainMethod(ml::ANN_MLP::BACKPROP, 0.0001);
    nn->setActivationFunction(ml::ANN_MLP::SIGMOID_SYM);
    nn->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 300, 0.0001));

    Mat train, test;
    Mat labels(1199, 2, CV_32F, 0.f); // 1399 - 200 test images
    for (size_t i = 0; i < fn.size(); i++) {
        // use the dnn as a "fixed function" preprocessor (no training here)
        Mat img = imread(fn[i]);
        net.setInput(dnn::blobFromImage(img, 1, inputImgSize, Scalar::all(127), false));
        Mat blob = net.forward("pool10");
        Mat f = blob.reshape(1, 1).clone(); // now our feature has 1000 numbers

        // sort into train/test slots:
        if (i < 100) {
            // test cat
            test.push_back(f);
        }
        else
            if (i >= 100 && i < 700) {
                // train cat
                train.push_back(f);
                labels.at<float>(i - 100, 0) = 1; // one-hot encoded labels for our ann
            }
            else
                if (i >= 700 && i < 800) {
                    // test dog
                    test.push_back(f);
                }
                else {
                    // train dog
                    train.push_back(f);
                    labels.at<float>(i - 200, 1) = 1;
                }
        cout << i << "\r"; // "machine learning should come with a statusbar." ;)
    }

    cout << train.size() << " " << labels.size() << " " << test.size() << endl;
    nn->train(train, 0, labels); // yes, that'll take a few minutes ..
    nn->save("cats.dogs.ann.yml.gz");

    Mat result;
    // our result array is one-hot endoded, too, which means:
    // if pin 1 is larger than pin 0, -  it predicted a dog, else a cat.
    nn->predict(test, result);
    // cout << result << endl;

    float correct_cat = 0;
    float correct_dog = 0;
    for (int i = 0; i < 100; i++) // 1st hundred testexamples were cats
        correct_cat += result.at<float>(i, 0) > result.at<float>(i, 1); // 0, true cat
    for (int i = 100; i < 200; i++) // 2nd hundred were dogs
        correct_dog += result.at<float>(i, 1) > result.at<float>(i, 0); // 1, true dog;
    float accuracy = (correct_cat + correct_dog) / 200;
    cout << correct_cat << " " << correct_dog << " : " << accuracy << endl;
    // 100 99 : 0.995

    return 0;
}
