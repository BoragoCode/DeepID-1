# DeepID
face verification

## Requirement
1. python3.6
2. pytorch

## Prepare Data
1. Download database from [Labeled Faces in the Wild](http://vis-www.cs.umass.edu/lfw/index.html);
2. Extract `lfw-deepfunneled.tgz` to `data/`;
3. Run `detect_lfw()` to detect faces, saved as `data/lfw_detect.txt`;
5. Run `gen_labels()` to generate labels, saved as `data/lfw_labels.txt`;
6. Run `gen_classify()` to generate patches, saved as `data/lfw_classify/lfw_classify_{0~8}.txt`;
    1. Only generate 9x3 face patches: 9 positions, 3 scales;
    3. scale is set as 0.85, 1.0, 1.15;

    ![patches](/images/patches.png)
    
7. Run `gen_classify_verify_pairs()` to generate pair samples, saved as `data/lfw_classify_verify/lfw_classify_verify_{0~8}_{train/valid/test}.txt`;

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
    |-- lfw-names.txt               # name number

    |-- lfw_detect.txt              # filepath x1 y1 x2 y2 xx1 yy1 xx2 yy2 xx3 yy3 xx4 yy4 xx5 yy5
    \-- lfw_classify
        |-- lfw_classify_{0~8}.txt  # filepath x1 y1 x2 y2 label
    \-- lfw_classify_verify
        |-- lfw_classify_verify_{0~8}_{train/valid/test}.txt  
                                    # label filepath1 x11 y11 x21 y21 label1 filepath2 x12 y12 x22 y22 label2
```

## Details

## Reference
1. [Deep Learning Face Representation from Predicting 10,000 Classes](https://ieeexplore.ieee.org/document/6909640?tp=&arnumber=6909640&refinements%3D4291944822%26sortType%3Ddesc_p_Publication_Year%26ranges%3D2014_2014_p_Publication_Year%26pageNumber%3D284%26rowsPerPage%3D100=).
2. [Deep Learning Face Representation by Joint Identification-Verification](https://www.researchgate.net/publication/263237688_Deep_Learning_Face_Representation_by_Joint_Identification-Verification)
3. [Deeply learned face representations are sparse, selective, and robust](http://xueshu.baidu.com/s?wd=paperuri:%28107f066157bf469540490ec52b0e65cd%29&filter=sc_long_sign&tn=SE_xueshusource_2kduw22v&sc_vurl=http://ieeexplore.ieee.org/xpls/icp.jsp?arnumber=7298907&ie=utf-8&sc_us=5787535078988981639)
4. [DeepID1 DeepID2 DeepID2+ DeepID3](https://blog.csdn.net/yuanchheneducn/article/details/51034463)
