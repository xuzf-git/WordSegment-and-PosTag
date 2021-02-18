# -*- encoding: utf-8 -*-
import json
import math
import time


class ShortTokenizer:
    def __init__(self, use_freq=True):
        self.word_freq = {}
        self.word_num = 0
        self.use_freq = use_freq

    def train(self, filepath, trained=False):
        """根据训练语料统计词频

        Args:
            filepath (string): 训练语料文件路径
            trained (bool): 模型是否已经训练
        """
        if not trained:
            # 统计词频
            print("正在训练模型……")
            stime = time.thread_time()
            with open(filepath, 'r', encoding='utf8') as f:
                for line in f.readlines():
                    line = line.strip().split()
                    self.word_num += len(line)
                    self.word_freq.update(
                        {i: self.word_freq.get(i, 0) + 1
                         for i in line})
            etime = time.thread_time()
            print("训练完成，耗时{}s".format(etime - stime))
            # 保存词频
            jsonstr = json.dumps(self.word_freq, ensure_ascii=False, indent=4)
            with open('./data/word_freq_npath.json', 'w',
                      encoding='utf8') as f:
                f.write(jsonstr)
        else:
            # 读入词频
            with open(filepath, 'r', encoding='utf8') as f:
                jsonstr = ''.join(f.readlines())
                self.word_freq = json.loads(jsonstr)
                self.word_num = sum(self.word_freq.values())

    def __weight(self, word):
        """计算word的词频 -log(P(w)) = log(num) - log(k_w)

        Args:
            word (string): 切分的词语，切分图上的一条边

        Returns:
            float: 词典中存在该词返回 -log(P)，否则返回0
        """
        freq = self.word_freq.get(word, 0)
        if freq and self.use_freq:
            return math.log(self.word_num) - math.log(freq)
        elif freq:
            return 1
        else:
            return 0

    def Token(self, sentence):
        """结合统计信息的最短路分词函数（最大概率分词）

        Args:
            sentence (string): 待切分的句子

        Returns:
            list: 切分的词语，构成的 list
        """
        length = len(sentence)
        # 构造句子的切分图
        graph = {}
        for i in range(length):
            graph[i] = []
            for j in range(i):
                freq = self.__weight(sentence[j:i + 1])
                if freq:
                    graph[i].append((j, freq))
        # 动态规划求解最优路径 ( arg min[-log(P)] )
        # 初始化DP矩阵
        dp = [(i, self.__weight(sentence[i])) for i in range(length)]
        dp.insert(0, (-1, 0))
        # 状态转移函数：dp[i] = min{dp[j-1] + weight(sentence[j:i])}
        # i：为当前词的词尾；j: 为当前词的词头
        for i in range(2, len(dp)):
            index = dp[i][0]
            cost = dp[i][1] + dp[i - 1][1]
            for j, freq in graph[i - 1]:
                if freq + dp[j][1] < cost:
                    cost = freq + dp[j][1]
                    index = j
            dp[i] = (index, cost)
        # 回溯最优路径
        token_res = []
        break_p = length
        while break_p > 0:
            token_res.append(sentence[dp[break_p][0]:break_p])
            break_p = dp[break_p][0]
        token_res.reverse()
        return token_res


# if __name__ == "__main__":
#     Tokenizer = ShortTokenizer()
#     # Tokenizer.train('./data/PeopleDaily_Token.txt')
#     Tokenizer.train('./data/word_freq_npath.json', trained=True)
#     Tokenizer.Token('迈向充满希望的新世纪')
#     Tokenizer.Token('１９９７年，是中国发展历史上非常重要的很不平凡的一年。')