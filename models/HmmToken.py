# -*- encoding: utf-8 -*-
import time
import json
import pandas as pd


class Hmm:
    def __init__(self):
        self.trans_p = {'S': {}, 'B': {}, 'M': {}, 'E': {}}
        self.emit_p = {'S': {}, 'B': {}, 'M': {}, 'E': {}}
        self.start_p = {'S': 0, 'B': 0, 'M': 0, 'E': 0}
        self.state_num = {'S': 0, 'B': 0, 'M': 0, 'E': 0}
        self.state_list = ['S', 'B', 'M', 'E']
        self.line_num = 0
        self.smooth = 1e-6

    @staticmethod
    def __state(word):
        """获取词语的BOS标签，标注采用 4-tag 标注方法，
        tag = {S,B,M,E}，S表示单字为词，B表示词的首字，M表示词的中间字，E表示词的结尾字

        Args:
            word (string): 函数返回词语 word 的状态标签
        """
        if len(word) == 1:
            state = ['S']
        else:
            state = list('B' + 'M' * (len(word) - 2) + 'E')
        return state

    def train(self, filepath, save_model=False):
        """训练hmm, 学习发射概率、转移概率等参数

        Args:
            save_model: 是否保存模型参数
            filepath (string): 训练预料的路径
        """
        print("正在训练模型……")
        start_time = time.thread_time()
        with open(filepath, 'r', encoding='utf8') as f:
            for line in f.readlines():
                self.line_num += 1
                line = line.strip().split()
                # 获取观测（字符）序列
                char_seq = list(''.join(line))
                # 获取状态（BMES）序列
                state_seq = []
                for word in line:
                    state_seq.extend(self.__state(word))
                # 判断是否等长
                assert len(char_seq) == len(state_seq)
                # 统计参数
                for i, s in enumerate(state_seq):
                    self.state_num[s] = self.state_num.get(s, 0) + 1.0
                    self.emit_p[s][char_seq[i]] = self.emit_p[s].get(
                        char_seq[i], 0) + 1.0
                    if i == 0:
                        self.start_p[s] += 1.0
                    else:
                        last_s = state_seq[i - 1]
                        self.trans_p[last_s][s] = self.trans_p[last_s].get(
                            s, 0) + 1.0
        # 归一化：
        self.start_p = {
            k: (v + 1.0) / (self.line_num + 4)
            for k, v in self.start_p.items()
        }
        self.emit_p = {
            k: {w: num / self.state_num[k]
                for w, num in dic.items()}
            for k, dic in self.emit_p.items()
        }
        self.trans_p = {
            k1: {k2: num / self.state_num[k1]
                 for k2, num in dic.items()}
            for k1, dic in self.trans_p.items()
        }
        end_time = time.thread_time()
        print("训练完成，耗时 {:.3f}s".format(end_time - start_time))
        # 保存参数
        if save_model:
            parameters = {
                'start_p': self.start_p,
                'trans_p': self.trans_p,
                'emit_p': self.emit_p
            }
            jsonstr = json.dumps(parameters, ensure_ascii=False, indent=4)
            param_filepath = "./data/HmmParam_Token.json"
            with open(param_filepath, 'w', encoding='utf8') as jsonfile:
                jsonfile.write(jsonstr)

    def viterbi(self, text):
        """Viterbi 算法

        Args:
            text (string): 句子

        Returns:
            list: 最优标注序列
        """
        text = list(text)
        dp = pd.DataFrame(index=self.state_list)
        # 初始化 dp 矩阵 (prop，last_state)
        dp[0] = [(self.start_p[s] * self.emit_p[s].get(text[0], self.smooth),
                  '_start_') for s in self.state_list]
        # 动态规划地更新 dp 矩阵
        for i, ch in enumerate(text[1:]):  # 遍历句子中的每个字符 ch
            dp_ch = []
            for s in self.state_list:  # 遍历当前字符的所有可能状态
                emit = self.emit_p[s].get(ch, self.smooth)
                # 遍历上一个字符的所有可能状态，寻找经过当前状态的最优路径
                (prob, last_state) = max([
                    (dp.loc[ls, i][0] * self.trans_p[ls].get(s, self.smooth) *
                     emit, ls) for ls in self.state_list
                ])
                dp_ch.append((prob, last_state))
            dp[i + 1] = dp_ch
        # 回溯最优路径
        path = []
        end = list(dp[len(text) - 1])
        back_point = self.state_list[end.index(max(end))]
        path.append(back_point)
        for i in range(len(text) - 1, 0, -1):
            back_point = dp.loc[back_point, i][1]
            path.append(back_point)
        path.reverse()
        return path

    def cut(self, text):
        """根据 viterbi 算法获得状态，根据状态切分句子

        Args:
            text (string): 待分词的句子

        Returns:
            list: 分词列表
        """
        state = self.viterbi(text)
        cut_res = []
        begin = 0
        for i, ch in enumerate(text):
            if state[i] == 'B':
                begin = i
            elif state[i] == 'E':
                cut_res.append(text[begin:i + 1])
            elif state[i] == 'S':
                cut_res.append(text[i])
        return cut_res


# if __name__ == "__main__":
#     hmm = Hmm()
#     hmm.train('./data/PeopleDaily_Token.txt', save_model=True)
#     cutres = hmm.cut('中央电视台收获一批好剧本')
#     print(cutres)
