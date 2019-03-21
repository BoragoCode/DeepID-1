import os

get_name_from_filepath = lambda filepath: filepath.split('/')[-2]

def gen_labels(datadir):
    detect_txt = os.path.join('/'.join(datadir.split('/')[:-1]), 'lfw_detect.txt')
    with open(detect_txt, 'r') as f:
        detect_list = f.readlines()
    
    # 读取所有检测出人脸的图片对应人名
    names = []
    for detect in detect_list:
        imgpath = detect.split(' ')[0]
        name = get_name_from_filepath(imgpath)
        names += [name]
    # 去重，产生对应标签
    names = list(set(names))
    labels = [i for i in range(len(names))]
    
    # 保存标签
    labels_txt = os.path.join('/'.join(datadir.split('/')[:-1]), 'lfw_labels.txt')
    f = open(labels_txt, 'w')
    for name, label in zip(names, labels):
        name_label = name + ' ' + str(label) + '\n'
        f.write(name_label)
    f.close()