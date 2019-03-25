import collections
import numpy as np

def read_data(path):
    dataset = open(path,'r')
    sentences = []
    sentence = ''
    for w in dataset:
        if w == '\n':
            sentences.append(sentence)
            sentence = ''
        else:
            sentence += w
    return dataset, sentences

def create_prob(sentences):
    wcounter = {}
    tcounter = {}
    emission = collections.defaultdict(dict)
    transition = collections.defaultdict(dict)
    for sentence in sentences:
        tokens = sentence.split('\n')
        pretag = 'start'
        for i,x in enumerate(tokens):
            if x:
                word_tag = x.split('\t')
                word = word_tag[0]
                tag = word_tag[1]
                wcounter[word] = wcounter.get(word,0)+1
                tcounter[tag] = tcounter.get(tag,0)+1
                emission[tag][word] = emission[tag].get(word,0)+1
                transition[pretag][tag] = transition[pretag].get(tag,0)+1
                pretag = tag        
    for x in transition:
        count = sum(transition[x].values())
        for y in transition[x]:
            transition[x][y] = np.log10(transition[x][y]/count)
    for x in emission:
        count = sum(emission[x].values())
        for y in emission[x]:
            emission[x][y] = np.log10(emission[x][y]/count)
    return wcounter,tcounter,transition,emission

def HandleUnknown(pre, word):
    n = len(word)
    pre = pre.lower()
    if word[0].isupper():
        tag = "NNP"
    elif pre in ["be","being","is","are","very","unlikely","extremely","so"]:
        tag = "JJ"
    elif pre in ["would","should","could","can","may"]:
        tag = "VB"
    elif pre == "it":
        tag = "VBZ"
    elif word[-2:] == "ss":
        tag = "NN"
    elif n >= 3 and (word[-3:] == "ble" or word[-3:] == "ive" or word[-2:] == "us"):
        tag ="JJ"
    elif word[-1:] == "s":
        tag = "NNS"
    elif "-" in word:
        tag = "JJ"
    elif word.isdigit():
        tag = "CD"
    elif "." in word:
        tag = "CD"
    elif pre == "$":
        tag = "CD"
    else:
        tag = "NN"
    return tag

def viterbi(words):
    n = len(words)
    dp = {}
    dp['start'] = (0,'start')
    for i,word in enumerate(words):
        new_dp = {}
        prew = words[i-1] if i>0 else ''
        unknowntag = HandleUnknown(prew, word)
        for pretag in dp:
            flag = True
            for tag in transition[pretag]:
                if word in emission[tag]:
                    flag = False
                    prob = dp[pretag][0]+transition[pretag][tag]+emission[tag][word]
                    new_path = dp[pretag][1]+' '+tag
                    if prob > new_dp.get(tag,(-float('inf'),''))[0]:
                        new_dp[tag] = (prob,new_path)
            if flag and unknowntag in transition[pretag]:
                prob = dp[pretag][0]+transition[pretag][unknowntag] - 10
                new_path = dp[pretag][1]+' '+unknowntag
                if prob > new_dp.get(unknowntag,(-float('inf'),''))[0]:
                        new_dp[unknowntag] = (prob,new_path)
        if not new_dp:
            new_dp[unknowntag] = (0,max(dp.values())[1]+' '+unknowntag)
        dp = new_dp
    return max(dp.values())

def write_output(path,sentences):
    file = open(path,'w')
    for sentence in sentences:
        words = sentence.split('\n')
        words.pop()
        tags = viterbi(words)[1].split(' ')
        for i in range(len(words)):
            file.write(words[i]+'\t'+tags[i+1]+'\n')
        file.write('\n')
    file.close()

training, sentences = read_data("./WSJ_POS_CORPUS_FOR_STUDENTS/WSJ_02-21.pos")

wcounter,tcounter,transition,emission = create_prob(sentences)

test, s = read_data("./WSJ_POS_CORPUS_FOR_STUDENTS/WSJ_24.words")

write_output("./WSJ_24.pos",s)
