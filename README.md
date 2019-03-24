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
5. Run `gen_classify()` to generate classify dataset, saved as `data/lfw_classify/lfw_classify.txt`;
6. Run `gen_classify_pairs()` to generate pair samples, saved as `data/lfw_classify_pairs/{train/valid/test}.txt`;

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
        |-- lfw_classify.txt                    # filepath x1 y1 x2 y2 xx1 yy1 xx2 yy2 xx3 yy3 xx4 yy4 xx5 yy5 label
    \-- lfw_classify_pairs
        |-- {train/valid/test}.txt              # filepath x1 y1 x2 y2 xx1 yy1 xx2 yy2 xx3 yy3 xx4 yy4 xx5 yy5 label\n
                                                # filepath x1 y1 x2 y2 xx1 yy1 xx2 yy2 xx3 yy3 xx4 yy4 xx5 yy5 label
```

## Load Data
1. ClassifyDataset(patch, scale)
    - Generate patches according to given parameter.
    - Returns one image and its label.

2. ClassifyPairsDataset(patch, scale, mode)
    - Generate patches according to given parameter.
    - Returns two images and their labels.

3. DeepIdDataset(mode)
    - Generate patches.
        - 9 positions
        - 3 scales
            set as `0.65, 1.0, 1.35`
    - Returns 9 patches, each patch contains 3 scales.

![patches](/images/patches.png)
    

## Models

![classifier_model](/images/classifier_model.png)

![verifier_model](/images/verifier_model.png)


1. Features model
    1. Consists of `conv1`, `conv2` and `features`;
        - `convx` is composed of `convolution layers`
        - `features` is a `fully connected later`
    3. the output of `maxpool3` and `conv4` will be used to generate features for verification;

    ```
    # input:    (batch, H, W, C(3))
    # output:   (batch, 160)
    ```

2. Classifier model
    Consists of `classifier`, which is a `connected later`;

    ```
    # input:    (batch, 160)
    # output:   (batch, n_class)
    ```

3. Model for classification
    Composed of `Features model` and `Classifier model`.
    
    **We need to train a model for each patch, which means 27 classification models.**

    ```
    # input:    (batch, H, W, C(3))
    # output:   (batch, n_class)
    ```

4. Verifier model
    1. Consists of `features` and `classifier`;
        - `features` is a dict
            - the keys are `classify_patch{}_scale{}`
            - the values are `locally connected layer`
        - `classifier` is a `fully connected layer`

    ```
    # input:    (batch, n_patch(27), 160*2)
    # output:   (batch)
    ```

3. DeepID model
    Consists of 27 `Features models` and a `Verifier`;
    - `Features models` are trained for classification task;

    **We need to generate 27 patches to feed the model.**

    ```
    # input:    (batch, n_sample(2), n_scale(3), n_channel(3), H, W) x 9
    # output:   (batch)
    ```

## Loss
1. Classification loss(Cross Entropy Loss)
    $$
    \text{loss}(x, class) = -\log\left(\frac{\exp(x[class])}{\sum_j \exp(x[j])}\right)
                    = -x[class] + \log\left(\sum_j \exp(x[j])\right)
    $$
    
2. Verification loss(BCE Loss)
    $$
    \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
    l_n = - w_n \left[ y_n \cdot \log x_n + (1 - y_n) \cdot \log (1 - x_n) \right]
    $$

3. Similarity loss, **Used to fine-tune the classification models.**
    **However, it seems useless.**
    $$
    \text{loss}(f_i, f_j, y_{ij}, \theta_{ve}) = 
    \begin{cases}
        \frac{1}{2} ||f_i - f_j||_2^2 & if y_{ij} =  1 \\
        \frac{1}{2} max(0, m-||f_i - f_j||_2)^2 & if y_{ij} = -1 
    \end{cases}
    $$

    where $m$ is the parameter to be learned.

## Details
1. Train 27 classify models first;
<!-- 2. Flip patches to generate more features; -->
2. We can set different learning rate to `DeepID` model's modules.
    1. set `Features` models' learning rate to 0.0
        **However, accuracy of testing is 42.79%. :-(**
    2. set lr to 0.01

## Reference
1. [Deep Learning Face Representation from Predicting 10,000 Classes](https://ieeexplore.ieee.org/document/6909640?tp=&arnumber=6909640&refinements%3D4291944822%26sortType%3Ddesc_p_Publication_Year%26ranges%3D2014_2014_p_Publication_Year%26pageNumber%3D284%26rowsPerPage%3D100=).
2. [Deep Learning Face Representation by Joint Identification-Verification](https://www.researchgate.net/publication/263237688_Deep_Learning_Face_Representation_by_Joint_Identification-Verification)
3. [Deeply learned face representations are sparse, selective, and robust](http://xueshu.baidu.com/s?wd=paperuri:%28107f066157bf469540490ec52b0e65cd%29&filter=sc_long_sign&tn=SE_xueshusource_2kduw22v&sc_vurl=http://ieeexplore.ieee.org/xpls/icp.jsp?arnumber=7298907&ie=utf-8&sc_us=5787535078988981639)
4. [DeepID1 DeepID2 DeepID2+ DeepID3](https://blog.csdn.net/yuanchheneducn/article/details/51034463)
