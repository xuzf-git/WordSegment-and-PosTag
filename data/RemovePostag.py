# -*- encoding: utf-8 -*-

str1 = ''

with open('./data/PeopleDaily_clean.txt', 'r', encoding='utf8') as f:
    for line in f.readlines():
        line = line.strip().split()
        line_clean = []
        for w in line:
            line_clean.append(w[:-2])
        str1 = str1 + ' '.join(line_clean) + '\n'

with open('./data/PeopleDaily_Token.txt', 'w', encoding='utf8') as f:
    f.write(str1)
