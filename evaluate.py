from models import ShortTokenizer
from models import HmmToken
from models import HmmPosTag
from utils import *
from tqdm import tqdm
import time
import os


def word_segmentation_eval(trainfile):
    """评估分词模型

    Args:
        trainfile (string): 训练数据文件路径

    Returns:
        list: 分词结果
    """
    with open(trainfile, 'r', encoding='utf8') as f:
        dataset = [line.strip().split(' ') for line in f.readlines()]
    dataset = dataset[0:6000]
    input_data = [''.join(line) for line in dataset]
    dataset_size = float(os.path.getsize(trainfile)) / 1024  # 以 kb 为单位

    # 利用 HMM 模型进行分词
    hmm_model = HmmToken.Hmm()
    hmm_model.train(trainfile, save_model=True)
    token_res = []
    print("HMM 分词模型：")
    stime = time.thread_time()
    for line in tqdm(input_data):
        token_res.append(hmm_model.cut(line))  # 预测分词
    etime = time.thread_time()
    evalutate(dataset, token_res)
    print("效率:\t{:.3f} kb/s\n".format(dataset_size / (etime - stime)))

    # 利用最短路分词模型
    st_model = ShortTokenizer.ShortTokenizer(use_freq=False)
    st_model.train(trainfile)
    token_res = []
    print("最短路分词模型：")
    stime = time.thread_time()
    for line in tqdm(input_data):
        token_res.append(st_model.Token(line))  # 预测分词
    etime = time.thread_time()
    evalutate(dataset, token_res)
    print("效率:\t{:.3f} kb/s\n".format(dataset_size / (etime - stime)))

    # 利用最大概率分词模型分词
    st_model = ShortTokenizer.ShortTokenizer(use_freq=True)
    st_model.train(trainfile)
    token_res = []
    print("最大概率分词模型：")
    stime = time.thread_time()
    for line in tqdm(input_data):
        token_res.append(st_model.Token(line))  # 预测分词
    etime = time.thread_time()
    evalutate(dataset, token_res)
    print("效率:\t{:.3f} kb/s\n".format(dataset_size / (etime - stime)))

    # 保存分词结果
    with open('./data/PeopleDaily_Token_hmm_result.txt', 'w', encoding='utf8') as f:
        for i in token_res:
            f.write(' '.join(i) + '\n')
    return token_res

def posTag_eval(trainfile, testfile):
    """评估词性标注模型

    Args:
        trainfile (string): 训练数据集路径
        testfile (string): 测试数据集路径

    Returns:
        list: 词性标注结果
    """
    hmm_pos = HmmPosTag.HmmPosTag()
    hmm_pos.train(trainfile)
    posTag_res = []
    dataset_size = float(os.path.getsize(testfile)) / 1024  # 以 kb 为单位
    with open(trainfile, 'r', encoding='utf8') as f:
        dataset = [line.strip().split(' ') for line in f.readlines()[:2000]]
    with open(testfile, 'r', encoding='utf8') as f:
        print("HMM 词性标注模型：")
        stime = time.thread_time()
        for line in tqdm(f.readlines()[:2000]):
            posTag_res.append(hmm_pos.predict(line.strip()))  # 预测分词
        etime = time.thread_time()
    evalutate(dataset, posTag_res)
    print("效率:\t{:.3f} kb/s\n".format(dataset_size / (etime - stime)))
    return posTag_res

if __name__ == "__main__":
    # 评估分词模型
    token_res = word_segmentation_eval('./data/PeopleDaily_Token.txt')
    # 评估词性标注
    # 在标准分词集合上标注词性
    trainfile = './data/PeopleDaily_clean.txt'
    testfile = './data/PeopleDaily_Token.txt'
    posTag_eval(trainfile, testfile)
    # 在最大概率分词集合上标注词性
    testfile = './data/PeopleDaily_Token_shortpath_result.txt'
    posTag_eval(trainfile, testfile)
    


