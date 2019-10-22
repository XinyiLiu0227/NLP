#!/usr/bin/python3
def read_weights(path):
    dataset = open(path,'r')
    preterminal = {}
    rule = {}
    first = {}
    last = {}
    terminal = set()
    for line in dataset:
        line = line.strip('\n')
        terminal.add(line[2])
        if line[0] == 'T':
            if line[2] not in preterminal:
                preterminal[line[2]] = {}
            preterminal[line[2]][line[4]] = int(line.split(' ')[-1])
        if line[0] == 'R':
            terminal.add(line[4])
            terminal.add(line[6])
            if line[2] not in rule:
                rule[line[2]] = {}
            rule[line[2]][line[4:7]] = int(line.split(' ')[-1])
        if line[0] == 'F':
            if line[2] not in first:
                first[line[2]] = {}
            first[line[2]][line[4]] = int(line.split(' ')[-1])
        if line[0] == 'L':
            if line[2] not in last:
                last[line[2]] = {}
            last[line[2]][line[4]] = int(line.split(' ')[-1])
    return terminal, preterminal, rule, first, last      

def read_data(path):
    dataset = open(path,'r')
    sentences = []
    for line in dataset:
        line = line.strip('\n')
        sentences.append(line)
    return sentences

def viterbi(sentence):
    sentence = sentence.split(' ')
    dp = {}
    n = len(sentence)
    for i in range(n):
        for tag in terminal:
            if tag not in dp:
                dp[tag] = {}
            dp[tag][i] = {}
            if tag in preterminal:
                if sentence[i] in preterminal[tag]:
                        dp[tag][i][i+1] = (preterminal[tag][sentence[i]], '('+tag+' '+sentence[i]+')')
                else:
                    dp[tag][i][i+1] = (0, '('+tag+' '+sentence[i]+')')
            if i+1 not in dp[tag][i] and tag!='S':
                dp[tag][i][i+1] = (0, '('+tag+' '+sentence[i]+')')
    for span in range(2, n+1):
        for i in range(n+1-span):
            for j in range(i+1, i+span):
                for root in sorted(terminal):
                    for left in sorted(terminal):
                        for right in sorted(terminal):
                            if not root in dp:
                                dp[root] = {}
                            if not i in dp[root]:
                                dp[root][i] = {}
                            key = str(left)+'_'+str(right)
                            temp = rule[root][key] if root in rule and key in rule[root] else 0
                            if root in first:
                                if sentence[i] in first[root]:
                                        temp += first[root][sentence[i]]
                            if root in last:
                                if sentence[i+span-1] in last[root]:
                                        temp += last[root][sentence[i+span-1]]
                            if left not in dp:
                                dp[left] = {}
                            if i not in dp[left]:
                                dp[left][i] = {}
                            if j not in dp[left][i]:
                                dp[left][i][j] = (-float('inf'), '(' + left+' '+sentence[i] + ')') 
                            temp += dp[left][i][j][0]
                            if right not in dp:
                                dp[right] = {}
                            if j not in dp[right]:
                                dp[right][j] = {}
                            if i+span not in dp[right][j]:
                                dp[right][j][i+span] = (-float('inf'), '(' + right+' '+sentence[j] + ')')
                            temp += dp[right][j][i+span][0]
                            if temp > dp[root][i].get(i+span, (-float('inf'), ''))[0]:
                                dp[root][i][i+span] = (temp, '(' + root + ' ' + dp[left][i][j][1] +' '+ dp[right][j][i+span][1] + ')')
    return dp['S'][0][n][0], dp['S'][0][n][1]

import argparse
parser = argparse.ArgumentParser(description='Parser.')
parser.add_argument('--data_file', type=str, default='test', help='test data file.')
parser.add_argument('--weights_file', type=str, default='weights', help='weight file.')
args = parser.parse_args()

terminal, preterminal, rule, first, last = read_weights(args.weights_file)
sentences = read_data(args.data_file)
for sentence in sentences:
    print(viterbi(sentence))
