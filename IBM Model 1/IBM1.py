#!/usr/bin/python3
import numpy as np
def read_data(path, insert = True):
    dataset = open(path, 'r', encoding = "ISO-8859-1")
    sentences = []
    for line in dataset:
        line = line.strip('\n')
        line = line.split(' ')
        if insert:
            line.insert(0, 'NULL')
        sentences.append(line)
    return sentences

def e_step(training_eng, training_fra, test_eng, test_fra, probability, t_table):
    print('begin e-step')
    for english, franch in zip(training_eng, training_fra):
        m = len(franch)
        l = len(english)
        print(english, franch)
        for f in franch:
            total = 0
            for e in english:
                if e not in probability:
                    probability[e] = {}
                if f not in probability[e]:
                    probability[e][f] = 10**(-6)
                t = probability[e][f]
                total += t
                # print('prob lookup: ' + e + ' ' + f + ' ' + str(t) + ' ' + str(total))
            for e in english:
                t = probability[e][f]/total
                if e not in t_table:
                    t_table[e] = {}
                t_table[e][f] = t_table[e].get(f, 0) + t
    
    return probability, t_table        

def loglikelihood(englishs, franchs, probability):
    logprob = 0
    for english, franch in zip(englishs, franchs):
        m = len(franch)
        l = len(english)
        for f in franch:
            prob = 0
            for e in english:
                if e not in probability or f not in probability[e]:
                    prob += 10**(-6)/l
                else:
                    prob += probability[e][f]/l
            logprob += np.log(prob)
    return logprob   

def m_step(probability, t_table):
    print('begin m-step')
    for e in probability:
        total = 0
        for f in probability[e]:
            total += t_table[e][f]
        for f in probability[e]:
            probability[e][f] = t_table[e][f]/total
            # print(e + ' ' + f + ' ' + str(t_table[e][f]) + ' ' + str(probability[e][f]))
    return probability

def em_iterations(n_iterations, training_eng, training_fra, test_eng, test_fra):
    probability = {}
    train_logprob = []
    test_logprob = []
    for i in range(n_iterations):
        print('begin iteration '+ str(i))
        t_table = {}
        probability, t_table = e_step(training_eng, training_fra, test_eng, test_fra, probability, t_table)
        logprob = loglikelihood(training_eng, training_fra,probability)
        train_logprob.append(logprob)
        print('training corpus logprob', logprob)
        logprob = loglikelihood(test_eng, test_fra, probability)
        test_logprob.append(logprob)
        print('test corpus logprob', logprob)
        probability = m_step(probability, t_table)
    return train_logprob, test_logprob       

# def plot_logprob(train_logprob, test_logprob):
#     import matplotlib.pyplot as plt
#     plt.plot(range(len(train_logprob)), train_logprob, c='r')
#     plt.plot(range(len(test_logprob)), test_logprob, c='b')
#     plt.show()

import argparse
parser = argparse.ArgumentParser(description='Machine Translation')
parser.add_argument('--train_eng', type=str, default='training.eng', help='English training data')
parser.add_argument('--train_fra', type=str, default='training.fra', help='French training data')
parser.add_argument('--test_eng', type=str, default='test.eng', help='English test data')
parser.add_argument('--test_fra', type=str, default='test.fra', help='French test data')
parser.add_argument('--num_iterations', type=int, default=5, help='the number of iterations')
args = parser.parse_args()

training_eng = read_data(args.train_eng)
training_fra = read_data(args.train_fra, insert = False)
test_eng = read_data(args.test_eng)
test_fra = read_data(args.test_fra, insert = False)

train_logprob, test_logprob = em_iterations(args.num_iterations, training_eng, training_fra, test_eng, test_fra)

#plot_logprob(train_logprob, test_logprob)

