#!/usr/bin/python3
def read_data(path):
    dataset = open(path,'r')
    trees = []
    sentences = []
    for line in dataset:
        line = line.strip('\n')
        trees.append(line)
        sentences.append([x for x in line if x.islower()])
    return trees, sentences

def viterbi(sentence):
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

def parse_nodes(tree):
    count = 0
    elements = []
    elements.append(tree[1])
    temp = []
    for x in tree[2:-1]:
        if x == '(':
            count += 1
        elif x == ')':
            count -= 1
        if x != ' ':
            temp.append(x)
        if count == 0 and temp:
            elements.append(temp)
            temp = []
    return elements[0], elements[1:]

def parse_tree(tree, weights):
    root, children = parse_nodes(tree)
    if len(children) > 1:
        left, first, _, weights = parse_tree(children[0], weights)
        right, _, last, weights  = parse_tree(children[1], weights)
        weights.append('R_' + root + '_' + left + '_' + right)
    else:
        weights.append('T_' + root + '_' + children[0][0])
        return root, children[0][0], children[0][0], weights

    weights.append('F_' + root + '_' + first)
    weights.append('L_' + root + '_' + last)
    return root, first, last, weights

def read_weights(terminal, preterminal, rule, first, last, weights, num):
    for line in weights:
        terminal.add(line[2])
        if line[0] == 'T':
            if line[2] not in preterminal:
                preterminal[line[2]] = {}
            preterminal[line[2]][line[4]] = preterminal[line[2]].get(line[4], 0) + num
        if line[0] == 'R':
            terminal.add(line[4])
            terminal.add(line[6])
            if line[2] not in rule:
                rule[line[2]] = {}
            rule[line[2]][line[4:7]] = rule[line[2]].get(line[4:7], 0) +  num
        if line[0] == 'F':
            if line[2] not in first:
                first[line[2]] = {}
            first[line[2]][line[4]] = first[line[2]].get(line[4], 0) + num
        if line[0] == 'L':
            if line[2] not in last:
                last[line[2]] = {}
            last[line[2]][line[4]] = last[line[2]].get(line[4], 0) + num
    return terminal, preterminal, rule, first, last

def run_iterations(terminal, preterminal, rule, first, last, n_iterations):
    for i in range(n_iterations):
        error = 0
        for sentence, tree in zip(sentences, trees):
            prob, predicted = viterbi(sentence)
            if predicted != tree:
                error += 1
                _, _, _, weights = parse_tree(list(tree), [])
                terminal, preterminal, rule, first, last = read_weights(terminal, preterminal, rule, first, last, weights, 1)
                _, _, _, weights = parse_tree(list(predicted), [])
                terminal, preterminal, rule, first, last = read_weights(terminal, preterminal, rule, first, last, weights, -1)
        if error == 0:
            return preterminal, rule, first, last
    return preterminal, rule, first, last


def write_output(preterminal, rule, first, last, path):
    file = open(path, 'w')
    for parent in preterminal:
        for key in preterminal[parent]:
            file.write('T_' + parent + '_' + key + ' ' + str(preterminal[parent][key]) + '\n')
    for parent in rule:
        for key in rule[parent]:
            file.write('R_' + parent + '_' + key + ' ' + str(rule[parent][key]) + '\n')
    for parent in first:
        for key in first[parent]:
            file.write('F_' + parent + '_' + key + ' ' + str(first[parent][key]) + '\n')
    for parent in last:
        for key in last[parent]:
            file.write('L_' + parent + '_' + key + ' ' + str(last[parent][key]) + '\n')     

import argparse
parser = argparse.ArgumentParser(description='Perceptron decoder.')
parser.add_argument('--data_file', type=str, default='train', help='training data file.')
parser.add_argument('--output_file', type=str, default='weights.learned', help='output file.')
args = parser.parse_args()

trees, sentences = read_data(args.data_file)
preterminal = {}
rule = {}
first = {}
last = {}
terminal = set()
weights = []
for tree in trees:
    _, _, _, weights = parse_tree(list(tree), weights)
terminal, preterminal, rule, first, last = read_weights(terminal, preterminal, rule, first, last, weights, 0)
preterminal, rule, first, last = run_iterations(terminal, preterminal, rule, first, last, 5)
write_output(preterminal, rule, first, last, args.output_file)

