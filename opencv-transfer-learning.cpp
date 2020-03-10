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
https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/squeezenet_v1.1.prototxt

https://github.com/opencv/opencv_extra/blob/master/testdata/dnn/squeezenet_v1.1.prototxt
https://github.com/yoggasek/Train_Data

https://answers.opencv.org/question/191359/ml-svm-k-nn-image-recognition-examples-in-c/

*/

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#include <fstream>
#include <map>
#include <numeric>

#include "filesystem.hpp"


using namespace cv;

namespace fs = std::filesystem;

auto getImagePaths(const std::string& location)
{
    std::map<std::string, std::vector<String>> result;

    for (const auto & folder : fs::directory_iterator(location))
    {
        //if (!folder.is_directory())
        //    continue;
        const auto name = folder.path().filename().string();

        std::vector<String> fn;
        glob(folder.path().string() + "/*.jpg", fn, true);
        result[name] = std::move(fn);
    }

    return result;
}

int main(int argc, char** argv)
{
    const auto trainingPaths = getImagePaths(argv[1] + std::string("/resized_for_training"));
    const auto testingPaths = getImagePaths(argv[1] + std::string("/resized_for_test"));

    const std::string modelTxt = "/data/mdl/squeezenet/squeezenet_v1.1.prototxt";
    const std::string modelBin = "/data/mdl/squeezenet/squeezenet_v1.1.caffemodel";

    std::vector<std::string> labelNames;
    for (const auto& l : trainingPaths)
    {
        labelNames.push_back(l.first);
    }

    dnn::Net net = dnn::readNetFromCaffe(modelTxt, modelBin);
    const cv::Size inputImgSize(227, 227); // model was trained with this size

    const char modelFileName[] = "cats.ann.yml.gz";

    const auto sampleCountingLambda = [](size_t n, const auto& v) { return n + v.second.size(); };

    Ptr<ml::ANN_MLP> nn;
    try {
        nn = ml::ANN_MLP::load(modelFileName);
    }
    catch (...) {
    }

    if (!nn)
    {
        nn = ml::ANN_MLP::create();

        Mat_<int> layers(4, 1);
        layers << 1000, 400, 80, trainingPaths.size(); // the sqeezenet pool10 layer has 1000 neurons

        nn->setLayerSizes(layers);
        nn->setTrainMethod(ml::ANN_MLP::BACKPROP, 0.0001);
        nn->setActivationFunction(ml::ANN_MLP::SIGMOID_SYM, 2.35, 1.0);
        nn->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 300, 0.00001));

        Mat train;

        const auto numLabels = std::accumulate(trainingPaths.begin(), trainingPaths.end(), 0u, sampleCountingLambda);

        Mat labels(numLabels, trainingPaths.size(), CV_32F, -1.f); // Stock activation function is symmetric, use -1 .. 1


        int categoryIdx = 0;
        int idx = 0;
        for (const auto& l : trainingPaths)
        {
            for (const auto& v : l.second)
            {
                // use the dnn as a "fixed function" preprocessor (no training here)
                Mat img = imread(v);
                net.setInput(dnn::blobFromImage(img, 1, inputImgSize, Scalar::all(127), false));
                Mat blob = net.forward("pool10");
                Mat f = blob.reshape(1, 1).clone(); // now our feature has 1000 numbers

                // train cat
                train.push_back(f);
                labels.at<float>(idx, categoryIdx) = 1; // one-hot encoded labels for our ann

                ++idx;
            }
            ++categoryIdx;
        }

        nn->train(train, 0, labels); // yes, that'll take a few minutes ..
        nn->save(modelFileName);
    }

    std::ofstream report("report.html");

    report << R"(<!DOCTYPE html>
<html lang="en">
  <head>
  </head>
  <body>
<h3>Test Data using github.com/freedomofkeima/transfer-learning-anime</h3>
)";

    float cumulativeError = 0;
    int categoryIdx = 0;
    for (const auto& l : testingPaths)
    {
        report << l.first << '\n';

        report << R"(
<table>
<thead>
<tr>
<th>Image</th>
<th>Classification</th>
<th>OK/NG</th>
</tr>
</thead>
<tbody>
)";

        Mat test;

        for (const auto& v : l.second)
        {
            // use the dnn as a "fixed function" preprocessor (no training here)
            Mat img = imread(v);
            net.setInput(dnn::blobFromImage(img, 1, inputImgSize, Scalar::all(127), false));
            Mat blob = net.forward("pool10");
            Mat f = blob.reshape(1, 1).clone(); // now our feature has 1000 numbers

            test.push_back(f);
        }

        Mat result;
        // our result array is one-hot endoded, too, which means:
        // if pin 1 is larger than pin 0, -  it predicted a dog, else a cat.
        nn->predict(test, result);

        int correct = 0;
        for (int i = 0; i < l.second.size(); ++i)
        {
            report << R"(<tr><td><a target="_blank" rel="noopener noreferrer" href=")"
                << l.second[i]
                << R"("><img src=")"
                << l.second[i]
                << R"(" alt="" style="max-width:100%;"></a></td><td>)";

            float h_max = -1.e20;
            int predicted = 0;
            for (int label = 0; label < trainingPaths.size(); ++label)
            {
                const auto h = (result.at<float>(i, label) + 1) / 2; // back to 0 .. 1
                const int expected = (label == categoryIdx) ? 1 : 0;

                cumulativeError += (h - expected) * (h - expected);

                if (label)
                    report << "<br>";
                report << labelNames[label] << " (score = " << h << ')';

                if (h > h_max)
                {
                    h_max = h;
                    predicted = label;
                }
            }
            const bool ok = predicted == categoryIdx;
            if (ok)
                ++correct;

            report << "</td><td>" << (ok? "OK" : "NG") << "</td></tr>\n";
        }

        report << "</tbody></table>\n";

        std::cout << "Score for " << l.first << " : " << correct << " : " << double(correct) / l.second.size() << '\n';

        ++categoryIdx;
    }

    report << "</body>\n</html>";

    const auto numTests = std::accumulate(testingPaths.begin(), testingPaths.end(), 0u, sampleCountingLambda);
    std::cout << "Average error: " << (cumulativeError / numTests) << '\n';

    return 0;
}
