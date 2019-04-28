import nltk
from nltk.classify import MaxentClassifier
from sklearn.cluster import KMeans

def WordEmbedding(path):
    dic = {}
    token_list = []
    vector_list = []
    lines = open(path, "r")
    for line in lines:
        word_vector = line.split(' ')
        dic[word_vector[0]] = [float(x) for x in word_vector[1:]]
        token_list.append(word_vector[0])
        vector_list.append(dic[word_vector[0]])
    return dic, token_list, vector_list
embeddings, token_list, vector_list = WordEmbedding('hw7/glove.6B.50d.txt')

def ReadData(path):
    lines = open(path, "r")
    startwords = []
    pretag = 'start'
    prepos = 'start'
    prechunk = 'start'
    features = []
    for line in lines:
        if line == '\n':
            pretag = 'start'
            prepos = 'start'
            prechunk = 'start'
        else:
            values = line.split()
            if pretag == 'start':
                startwords.append(values[0])
            else:
                features[-1][-1] = values[0]
            features.append([pretag,prepos,prechunk]+values+['end'])
            prepos = values[1]
            prechunk = values[2]
            pretag = values[3]
    return features, startwords
features,startwords = ReadData("CONLL_NAME_CORPUS_FOR_STUDENTS/CONLL_train.pos-chunk-name")


def Clustering(li, num):
    kmeans = KMeans(n_clusters=num, random_state=0).fit(li)
    return kmeans
kmeans = Clustering(vector_list,400)


def FeatureBuilder(pretag,prepos,prechunk,token,pos,chunk,nextword):
    feature = {}
    feature['token'] = token
    feature['pos'] = pos
    feature['chunk'] = chunk
    feature['cap'] = token[0].isupper()
    feature['startwords'] = token in startwords
    feature['cap_start'] = token not in startwords and token[0].isupper()
    feature['pretag'] = pretag
    feature['name_list1'] = "Jan" in token
    feature['name_list2'] = "Tom" in token
    feature['preposchunk'] = (prepos,prechunk)
    feature['nextword'] = nextword.isdigit()
    feature['binarization'] = []
    if token.lower() in embeddings:
        feature['cluster'] = kmeans.labels_[token_list.index(token.lower())]
        for num in embeddings[token.lower()]:
            if num>=0:
                feature['binarization'].append(1)
            else:
                feature['binarization'].append(0)
    else:
        feature['cluster'] = 401
        feature['binarization'] = [0]*50
    feature['binarization'] = tuple(feature['binarization'])
    return feature

def MEtrain(feature, iterations = 20):
    return MaxentClassifier.train(feature, max_iter=iterations)
MEMM = MEtrain([(FeatureBuilder(pretag,prepos,prechunk,token,pos,chunk,nextword),tag) for pretag,prepos,prechunk,token,pos,chunk,tag,nextword in features])

def ReadTestData(path):
    lines = open(path, "r")
    tokens = []
    pos = []
    chunk = []
    words = []
    poslist = []
    chunklist = []
    for line in lines:
        if line == '\n':
            words.append(tokens)
            poslist.append(pos)
            chunklist.append(chunk)
            tokens = []
            pos = []
            chunk = []
        else:
            values = line.split()
            tokens.append(values[0])
            pos.append(values[1])
            chunk.append(values[2])
    return words, poslist,chunklist
words, poslist,chunklist = ReadTestData("CONLL_NAME_CORPUS_FOR_STUDENTS/CONLL_dev.pos-chunk")

def MEtag(tokens,pos,chunk):
    tags = ['I-PER','I-LOC','I-ORG','I-MISC','O']
    n = len(tokens)
    dp = {}
    dp['start'] = (1,[])
    for i,word in enumerate(tokens[:-1]):
        new_dp = {}
        for pretag in dp:
            probs = MEMM.prob_classify(FeatureBuilder(pretag,pos[i-1],chunk[i-1],word,pos[i],chunk[i],tokens[i+1]))
            for tag in tags:
                probability = probs.prob(tag)
                probability *= dp[pretag][0]
                new_path = dp[pretag][1]+[tag]
                if probability > new_dp.get(tag,(-float('inf'),''))[0]:
                    new_dp[tag] = (probability,new_path)
        dp = new_dp
    return max(dp.values())

def WriteOutput(path,words,poslist,chunklist):
    file = open(path,'w')
    for tokens,pos,chunk in zip(words,poslist,chunklist):
        tags = MEtag(tokens+['end'],pos+['start'],chunk+['start'])[1]
        for i in range(len(tokens)):
            file.write(tokens[i]+'\t'+tags[i]+'\n')
        file.write('\n')
    file.close()

WriteOutput('CONLL_dev.name',words,poslist,chunklist)

