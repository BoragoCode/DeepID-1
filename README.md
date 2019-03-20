# DeepID
face verification

## Requirement
1. python3.6
2. pytorch

## Prepare Data
1. Download database from [Labeled Faces in the Wild](http://vis-www.cs.umass.edu/lfw/index.html);
    - 13233 images
    - 5749 people
    - 1680 people with two or more images
2. Extract `lfw-deepfunneled.tgz` to `data/`;
3. Run `detect_lfw()` to detect faces, saved as `data/lfw_detect.txt`;
4. Run `gen_labels()` to generate labels, saved as `data/lfw_labels.txt`;
5. Run `gen_classify()` to generate patches, saved as `data/lfw_classify/lfw_classify_{0~8}.txt`;
    1. Only generate 9x3 face patches: 9 positions, 3 scales;
    2. scale is set as 0.85, 1.0, 1.15;
    3. save features(9x3x160) when testing;
        - 9 patches, 3 scales, 160 dimensions 

    ![patches](/images/patches.png)

6. Run `gen_verify_pairs()` to generate pair samples, saved as `lfw_verify/lfw_verify_{train/valid/test}.txt`;
7. Run `gen_classify_similarity_pairs()` to generate pair samples, saved as `data/lfw_similarity_verify/lfw_classify_similarity_{0~8}_{train/valid/test}.txt`;
8. Run `gen_deepid_pairs_samples()` to generate pair samples for the whole net, saved as `data/lfw_deepid_pair/lfw_deepid_pair_{train/valid/test}/txt`

```
\-- data
    \-- lfw-deepfunneled
        \-- {name}
            |-- {name}_{xxxx}.jpg
    \-- view1
        |-- pairsDevTest.txt
        |-- pairsDevTrain.txt
        |-- peopleDevTest.txt
        |-- peopleDevTrain.txt
    \-- view2
        |-- pairs.txt
        |-- people
    |-- lfw-names.txt                           # name number


    |-- lfw_detect.txt                          # filepath x1 y1 x2 y2 xx1 yy1 xx2 yy2 xx3 yy3 xx4 yy4 xx5 yy5
    |-- lfw_labels.txt                          # name label
    \-- lfw_classify
        |-- lfw_classify_{0~8}.txt              # filepath x1 y1 x2 y2 label
    \-- lfw_verify
        |-- lfw_verify_{train/valid/test}.txt   # index1, index2, label
    \-- lfw_classify_similarity
        |-- lfw_classify_similarity_{0~8}.txt   # filepath1 x11 y11 x21 y21 label1\n
                                                # filepath2 x12 y12 x22 y22 label2
    

    
    \-- features                                # features saved when testing classify models
        |-- lfw_classify_{0~8}_scale{S,M,L}.npy

```

## Models
1. Classify model
    1. Combined with `features` and `classifiers`;
    2. `features` contains `convolution layers` and a `connected later`;
    3. the output of `maxpool3` and `conv4` will be used in verification;

    ![classifier_model](/images/classifier_model.png)

2. Verify model
    1. Combined with `features` and `classifiers`;
    2. `features` is combined with `features` of `Classify model` and a `locally-connected layer`;
    3. `classifiers` contains a `fully-connected layer`

    ![verifier_model](/images/verifier_model.png)

## Loss
1. Classification loss(Cross Entropy)
$$
\text{loss}(x, class) = -\log\left(\frac{\exp(x[class])}{\sum_j \exp(x[j])}\right)
                   = -x[class] + \log\left(\sum_j \exp(x[j])\right)
$$
    
2. Verification loss(BCE)
$$
\ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
l_n = - w_n \left[ y_n \cdot \log x_n + (1 - y_n) \cdot \log (1 - x_n) \right]
$$

3. Similarity loss, **Used to fine-tune the classification models.**
$$
\text{loss}(f_i, f_j, y_{ij}, \theta_{ve}) = 
\begin{cases}
    \frac{1}{2} ||f_i - f_j||_2^2 & if y_{ij} =  1 \\
    \frac{1}{2} max(0, m-||f_i - f_j||_2)^2 & if y_{ij} = -1 
\end{cases}
$$

where $m$ is the parameter to be learned.

## Details
1. Train classify models first;
2. After trainig the classify models, save features as `/data/lfw_classify/lfw_classify_{patch}.npy`;
<!-- 3. Flip patches to generate more features; -->

## Reference
1. [Deep Learning Face Representation from Predicting 10,000 Classes](https://ieeexplore.ieee.org/document/6909640?tp=&arnumber=6909640&refinements%3D4291944822%26sortType%3Ddesc_p_Publication_Year%26ranges%3D2014_2014_p_Publication_Year%26pageNumber%3D284%26rowsPerPage%3D100=).
2. [Deep Learning Face Representation by Joint Identification-Verification](https://www.researchgate.net/publication/263237688_Deep_Learning_Face_Representation_by_Joint_Identification-Verification)
3. [Deeply learned face representations are sparse, selective, and robust](http://xueshu.baidu.com/s?wd=paperuri:%28107f066157bf469540490ec52b0e65cd%29&filter=sc_long_sign&tn=SE_xueshusource_2kduw22v&sc_vurl=http://ieeexplore.ieee.org/xpls/icp.jsp?arnumber=7298907&ie=utf-8&sc_us=5787535078988981639)
4. [DeepID1 DeepID2 DeepID2+ DeepID3](https://blog.csdn.net/yuanchheneducn/article/details/51034463)
