import random


def data_clean(infilepath, outfilepath):
    """数据清洗：去除空行、组合实体合并、人名合并、词类整合

    Args:
        infilepath (string): 未清洗的数据集文件路径
        outfilepath (string): 清洗后数据文件路径
    """
    infile = open(infilepath, 'r', encoding='utf-8')
    outfile = open(outfilepath, 'w', encoding='utf-8')
    for line in infile.readlines():
        line = line.split('  ')
        if line[0][0] != '1':
            continue
        i = 1
        while i < len(line) - 1:
            if line[i][0] == '[':  # 组合实体名
                word = line[i].split('/')[0][1:]
                i += 1
                while i < len(line) - 1 and line[i].find(']') == -1:
                    if line[i] != '':
                        word += line[i].split('/')[0]
                    i += 1
                word += line[i].split('/')[0].strip() + '/n '
            elif line[i].split('/')[1] == 'nr':  # 人名
                word = line[i].split('/')[0]
                i += 1
                if i < len(line) - 1 and line[i].split('/')[1] == 'nr':
                    word += line[i].split('/')[0] + '/n '
                else:
                    word += '/n '
                    i -= 1
            elif line[i].split('/')[1][0] == 'n':
                word = line[i].split('/')[0] + '/n '
            else:
                word = line[i].split('/')[0] + '/' 
                word += line[i].split('/')[1][0].lower() + ' '
            outfile.write(word)
            i += 1
        outfile.write('\n')
    infile.close()
    outfile.close()


def train_test_split(datafile, train_rate=0.8):
    """随机划分训练集和测试集

    Args:
        datafile (fileIOwrapper): 训练集文件指针
        train_rate (float, optional): 训练集占比. Defaults to 0.8.

    Returns:
        （list, list）: (train set, test set)
    """
    with open(datafile, 'r', encoding='utf8') as f:
        dataset = f.readlines()
    train_set_size = round(len(dataset) * train_rate)
    train_set = random.sample(dataset, train_set_size)
    test_set = list(set(dataset).difference(set(train_set)))
    return train_set, test_set


def eval(predict, groundtruth):
    """计算预测结果的准确率、召回率、F1

    Args:
        predict (list): 预测结果
        groundtruth (list): 真实结果

    Returns:
        tuple(precision, recall, f1): 精确率, 召回率, f1
    """
    assert len(predict) == len(groundtruth)
    tp, fp, tn, fn = 0, 0, 0, 0
    for i in range(len(predict)):
        right = len([j for j in predict[i] if j in groundtruth[i]])
        tp += right
        fn += len(groundtruth[i]) - right
        fp += len(predict[i]) - right
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def evalutate(dataset, token_res):
    """打印测试结果

    Args:
        dataset (list): 真实结果
        token_res (list): 分词结果
    """
    precision, recall, f1 = eval(token_res, dataset)
    print("精确率:\t{:.3%}".format(precision))
    print("召回率:\t{:.3%}".format(recall))
    print("f1:\t{:.3%}".format(f1))