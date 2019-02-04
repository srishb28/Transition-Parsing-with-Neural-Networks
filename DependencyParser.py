import collections
import tensorflow as tf
import numpy as np
import pickle
import math
from progressbar import ProgressBar

from DependencyTree import DependencyTree
from ParsingSystem import ParsingSystem
from Configuration import Configuration
import Config
import Util

"""
This script defines a transition-based dependency parser which makes
use of a classifier powered by a neural network. The neural network
accepts distributed representation inputs: dense, continuous
representations of words, their part of speech tags, and the labels
which connect words in a partial dependency parse.

This is an implementation of the method described in

Danqi Chen and Christopher Manning. A Fast and Accurate Dependency Parser Using Neural Networks. In EMNLP 2014.

Author: Danqi Chen, Jon Gauthier
Modified by: Heeyoung Kwon (2017)
Modified by: Jun S. Kang (2018 Mar)
"""


class DependencyParserModel(object):

    def __init__(self, graph, embedding_array, Config):

        self.build_graph(graph, embedding_array, Config)

    def build_graph(self, graph, embedding_array, Config):
        """

        :param graph:
        :param embedding_array:
        :param Config:
        :return:
        """

        with graph.as_default():
             #for fixed embeddings, trainable =False was added
             self.embeddings = tf.Variable(embedding_array, dtype=tf.float32)#trainable=False)

             """
            ===================================================================

            Define the computational graph with necessary variables.
            
            1) You may need placeholders of:
                - Many parameters are defined at Config: batch_size, n_Tokens, etc
                - # of transitions can be get by calling parsing_system.numTransitions()
                
            self.train_inputs = 
            self.train_labels = 
            self.test_inputs =
            ...
            
                
            2) Call forward_pass and get predictions
            
            ...
            self.prediction = self.forward_pass(embed, weights_input, biases_input, weights_output)


            3) Implement the loss function described in the paper
             - lambda is defined at Config.lam
            
            ...
            self.loss =
            
            ===================================================================
            """
             self.train_inputs = tf.placeholder(tf.int32, shape=(Config.batch_size, Config.n_Tokens))
             self.train_labels = tf.placeholder(tf.float32, shape=(Config.batch_size, parsing_system.numTransitions()))
             self.test_inputs = tf.placeholder(tf.int32, shape=(Config.n_Tokens))

             weights_input = tf.Variable(tf.truncated_normal([Config.hidden_size, Config.n_Tokens * Config.embedding_size], stddev=0.1))
             biases_input = tf.Variable(tf.zeros((Config.hidden_size,)))
             #weights_input1 = tf.Variable(tf.truncated_normal([Config.hidden_size,  Config.hidden_size1],stddev=0.1))
             #biases_input1 = tf.Variable(tf.zeros((Config.hidden_size1,)))
             #weights_input2 = tf.Variable(tf.truncated_normal([Config.hidden_size1,  Config.hidden_size2],stddev=0.1))
             #biases_input2 = tf.Variable(tf.zeros((Config.hidden_size2,)))
             #weights_output = tf.Variable(tf.truncated_normal([Config.hidden_size, parsing_system.numTransitions()], stddev=0.1))
             weights_output = tf.Variable(tf.truncated_normal([Config.hidden_size, parsing_system.numTransitions()], stddev=0.1))
             embed = tf.nn.embedding_lookup(self.embeddings, self.train_inputs)
             embed = tf.reshape(embed, [Config.batch_size, -1])

             self.prediction = self.forward_pass(embed, weights_input, biases_input, weights_output)

             cross_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.prediction,
                                                                                               labels=tf.argmax(
                                                                                                   self.train_labels,axis=1)))
             
             regularizer_term=tf.nn.l2_loss(embed)+tf.nn.l2_loss(weights_input)+tf.nn.l2_loss(biases_input)+tf.nn.l2_loss(weights_output)
             regularizer_term=Config.lam*0.5*regularizer_term

             self.loss = cross_loss + regularizer_term
            
             

             optimizer = tf.train.GradientDescentOptimizer(Config.learning_rate)
             #self.app=optimizer.compute_gradients(self.loss)
             grads = optimizer.compute_gradients(self.loss)
             clipped_grads = [(tf.clip_by_norm(grad, 5), var) for grad, var in grads]
             self.app = optimizer.apply_gradients(clipped_grads)

            # For test data, we only need to get its prediction
             test_embed = tf.nn.embedding_lookup(self.embeddings, self.test_inputs)
             test_embed = tf.reshape(test_embed, [1, -1])
             self.test_pred = tf.nn.softmax(self.forward_pass(test_embed, weights_input, biases_input, weights_output))

            # intializer
             self.init = tf.global_variables_initializer()
            
             
             
    def train(self, sess, num_steps):
        """

        :param sess:
        :param num_steps:
        :return:
        """
        self.init.run()
        print("Initailized")

        average_loss = 0
        for step in range(num_steps):
            start = (step * Config.batch_size) % len(trainFeats)
            end = ((step + 1) * Config.batch_size) % len(trainFeats)
            if end < start:
                start -= end
                end = len(trainFeats)
            batch_inputs, batch_labels = trainFeats[start:end], trainLabels[start:end]

            feed_dict = {self.train_inputs: batch_inputs, self.train_labels: batch_labels}

            _, loss_val = sess.run([self.app, self.loss], feed_dict=feed_dict)
            average_loss += loss_val

            if step % Config.display_step == 0:
                if step > 0:
                    average_loss /= Config.display_step
                print("Average loss at step ", step, ": ", average_loss)
                average_loss = 0
            if step % Config.validation_step == 0 and step != 0:
                print("\nTesting on dev set at step ", step)
                predTrees = []
                for sent in devSents:
                    numTrans = parsing_system.numTransitions()

                    c = parsing_system.initialConfiguration(sent)
                    while not parsing_system.isTerminal(c):
                        feat = getFeatures(c)
                        pred = sess.run(self.test_pred, feed_dict={self.test_inputs: feat})

                        optScore = -float('inf')
                        optTrans = ""

                        for j in range(numTrans):
                            if pred[0, j] > optScore and parsing_system.canApply(c, parsing_system.transitions[j]):
                                optScore = pred[0, j]
                                optTrans = parsing_system.transitions[j]

                        c = parsing_system.apply(c, optTrans)

                    predTrees.append(c.tree)
                result = parsing_system.evaluate(devSents, predTrees, devTrees)
                print(result)

        print("Train Finished.")

    def evaluate(self, sess, testSents):
        """

        :param sess:
        :return:
        """

        print("Starting to predict on test set")
        predTrees = []
        for sent in testSents:
            numTrans = parsing_system.numTransitions()

            c = parsing_system.initialConfiguration(sent)
            while not parsing_system.isTerminal(c):
                # feat = getFeatureArray(c)
                feat = getFeatures(c)
                pred = sess.run(self.test_pred, feed_dict={self.test_inputs: feat})

                optScore = -float('inf')
                optTrans = ""

                for j in range(numTrans):
                    if pred[0, j] > optScore and parsing_system.canApply(c, parsing_system.transitions[j]):
                        optScore = pred[0, j]
                        optTrans = parsing_system.transitions[j]

                c = parsing_system.apply(c, optTrans)

            predTrees.append(c.tree)
        print("Saved the test results.")
        Util.writeConll('result_test_bestmod.conll', testSents, predTrees)

    def forward_pass(self, embed, weights_input, biases_input, weights_output):
        """

        :param embed:
        :param weights:
        :param biases:
        :return:
        """
        """
        =======================================================

        Implement the forwrad pass described in
        "A Fast and Accurate Dependency Parser using Neural Networks"(2014)

        =======================================================
        """
        #basic model with different activation layers.
        input_layer = tf.matmul(embed, tf.transpose(weights_input))
        hidden_input = tf.pow(tf.add(input_layer, biases_input),3)
##        #hidden_input = tf.nn.relu(tf.add(input_layer, biases_input))
        #hidden_input = tf.sigmoid(tf.add(input_layer, biases_input))
        #hidden_input = tf.tanh(tf.add(input_layer, biases_input))
        output_layer = tf.matmul(hidden_input, weights_output)


         #  model with two hidden layers and different activation functions

#        input_layer=tf.matmul(embed,tf.transpose(weights_input))
#        hidden_layer1=tf.pow(tf.add(input_layer,biases_input),3)
##        hidden_layer1=tf.nn.relu(tf.add(input_layer,biases_input))   
#        hidden_layer2=tf.matmul(hidden_layer1,weights_input1)
#        hidden_layer2=tf.pow(tf.add(hidden_layer2,biases_input1),3)
##        hidden_layer2=tf.nn.relu(tf.add(hidden_layer2,biases_input1))
#        output_layer=tf.matmul(hidden_layer2,weights_output)
#        return output_layer


         # model with three hidden layers
#        input_layer=tf.matmul(embed,tf.transpose(weights_input))
#        hidden_layer1=tf.pow(tf.add(input_layer,biases_input),3)        
#        hidden_layer2=tf.matmul(hidden_layer1,weights_input1)
#        hidden_layer2=tf.pow(tf.add(hidden_layer2,biases_input1),3)
#        hidden_layer3=tf.matmul(hidden_layer2,weights_input2)
#        hidden_layer3=tf.pow(tf.add(hidden_layer3,biases_input2),3)      
#        output_layer=tf.matmul(hidden_layer3,weights_output)
#        return output_layer
         
         
         
         
         #basic model with parallel hidden layers for words, pos and arcs
         #Slicing Embed and Weigh Matrices to get parallel cubic activation function
#        embed1, embed2, embed3 = tf.split(embed,[18*Config.embedding_size,18*Config.embedding_size,12*Config.embedding_size],1)
#        w11, w12 , w13 = tf.split(weights_input,[18*Config.embedding_size,18*Config.embedding_size,12*Config.embedding_size],1)   
#        product1 = tf.add(tf.matmul(embed1,tf.transpose(w11)),biases_input)
#        product2= tf.add(tf.matmul(embed2,tf.transpose(w12)),biases_input)
#        product3= tf.add(tf.matmul(embed3,tf.transpose(w13)),biases_input)
#        hidden_layer1 = tf.pow(product1,3)
#        hidden_layer2 = tf.pow(product2,3)
#        hidden_layer3 = tf.pow(product3,3)
#        output_layer1 = tf.matmul(hidden_layer1,weights_output)
#        output_layer2=tf.add(tf.matmul(hidden_layer2,weights_output),output_layer1)
#        output_layer=tf.add(tf.matmul(hidden_layer3,weights_output),output_layer2)
#        return output_layer

        return output_layer
        


def genDictionaries(sents, trees):
    word = []
    pos = []
    label = []
    for s in sents:
        for token in s:
            word.append(token['word'])
            pos.append(token['POS'])

    rootLabel = None
    for tree in trees:
        for k in range(1, tree.n + 1):
            if tree.getHead(k) == 0:
                rootLabel = tree.getLabel(k)
            else:
                label.append(tree.getLabel(k))

    if rootLabel in label:
        label.remove(rootLabel)

    index = 0
    wordCount = [Config.UNKNOWN, Config.NULL, Config.ROOT]
    wordCount.extend(collections.Counter(word))
    for word in wordCount:
        wordDict[word] = index
        index += 1

    posCount = [Config.UNKNOWN, Config.NULL, Config.ROOT]
    posCount.extend(collections.Counter(pos))
    for pos in posCount:
        posDict[pos] = index
        index += 1

    labelCount = [Config.NULL, rootLabel]
    labelCount.extend(collections.Counter(label))
    for label in labelCount:
        labelDict[label] = index
        index += 1

    return wordDict, posDict, labelDict


def getWordID(s):
    if s in wordDict:
        return wordDict[s]
    else:
        return wordDict[Config.UNKNOWN]


def getPosID(s):
    if s in posDict:
        return posDict[s]
    else:
        return posDict[Config.UNKNOWN]


def getLabelID(s):
    if s in labelDict:
        return labelDict[s]
    else:
        return labelDict[Config.UNKNOWN]


def getFeatures(c):
    """
    =================================================================

    Implement feature extraction described in
    "A Fast and Accurate Dependency Parser using Neural Networks"(2014)

    =================================================================
    """

    words=[]
    labels=[]
    arcs=[]
    for i in range(0,3):
        words.append(getWordID(c.getWord(c.getStack(i))))
        labels.append(getPosID(c.getPOS(c.getStack(i))))
    for i in range(0,3):
        words.append(getWordID(c.getWord(c.getBuffer(i))))
        labels.append(getPosID(c.getPOS(c.getBuffer(i))))
    for i in range(0,2):
        words.append(getWordID(c.getWord(c.getLeftChild(c.getStack(i),1))))
        labels.append(getPosID(c.getPOS(c.getLeftChild(c.getStack(i),1))))
        arcs.append(getLabelID(c.getLabel(c.getLeftChild(c.getStack(i),1))))
        
        words.append(getWordID(c.getWord(c.getRightChild(c.getStack(i),1))))
        labels.append(getPosID(c.getPOS(c.getRightChild(c.getStack(i),1))))
        arcs.append(getLabelID(c.getLabel(c.getRightChild(c.getStack(i),1))))
        
        words.append(getWordID(c.getWord(c.getLeftChild(c.getStack(i),2))))
        labels.append(getPosID(c.getPOS(c.getLeftChild(c.getStack(i),2))))
        arcs.append(getLabelID(c.getLabel(c.getLeftChild(c.getStack(i),2))))
        
        words.append(getWordID(c.getWord(c.getRightChild(c.getStack(i),2))))
        labels.append(getPosID(c.getPOS(c.getRightChild(c.getStack(i),2))))
        arcs.append(getLabelID(c.getLabel(c.getRightChild(c.getStack(i),2))))
        
        words.append(getWordID(c.getWord(c.getLeftChild(c.getLeftChild(c.getStack(i),1),1))))
        labels.append(getPosID(c.getPOS(c.getLeftChild(c.getLeftChild(c.getStack(i),1),1))))
        arcs.append(getLabelID(c.getLabel(c.getLeftChild(c.getLeftChild(c.getStack(i),1),1))))
        
        
        words.append(getWordID(c.getWord(c.getRightChild(c.getRightChild(c.getStack(i),1),1))))
        labels.append(getPosID(c.getPOS(c.getRightChild(c.getRightChild(c.getStack(i),1),1))))
        arcs.append(getLabelID(c.getLabel(c.getRightChild(c.getRightChild(c.getStack(i),1),1))))
        
    features=[]
    features.extend(words)
    features.extend(labels)
    features.extend(arcs)
    return features


def genTrainExamples(sents, trees):
    numTrans = parsing_system.numTransitions()

    features = []
    labels = []
    pbar = ProgressBar()
    for i in pbar(range(len(sents))):
        if trees[i].isProjective():
            c = parsing_system.initialConfiguration(sents[i])

            while not parsing_system.isTerminal(c):
                oracle = parsing_system.getOracle(c, trees[i])
                feat = getFeatures(c)
                label = []
                for j in range(numTrans):
                    t = parsing_system.transitions[j]
                    if t == oracle:
                        label.append(1.)
                    elif parsing_system.canApply(c, t):
                        label.append(0.)
                    else:
                        label.append(-1.)

                if 1.0 not in label:
                    print(i, label)
                features.append(feat)
                labels.append(label)
                c = parsing_system.apply(c, oracle)
            #if(c.tree.equal(trees[i])):
             #   print("success")
    return features, labels


def load_embeddings(filename, wordDict, posDict, labelDict):
    dictionary, _, word_embeds = pickle.load(open(filename, 'rb'))

    embedding_array = np.zeros((len(wordDict) + len(posDict) + len(labelDict), Config.embedding_size))
    knownWords = list(wordDict.keys())
    foundEmbed = 0
    for i in range(len(embedding_array)):
        index = -1
        if i < len(knownWords):
            w = knownWords[i]
            if w in dictionary:
                index = dictionary[w]
            elif w.lower() in dictionary:
                index = dictionary[w.lower()]
        if index >= 0:
            foundEmbed += 1
            embedding_array[i] = word_embeds[index]
        else:
            embedding_array[i] = np.random.rand(Config.embedding_size) * 0.02 - 0.01
    print("Found embeddings: ", foundEmbed, "/", len(knownWords))

    return embedding_array


if __name__ == '__main__':

    wordDict = {}
    posDict = {}
    labelDict = {}
    parsing_system = None

    trainSents, trainTrees = Util.loadConll('train.conll')
    devSents, devTrees = Util.loadConll('dev.conll')
    testSents, _ = Util.loadConll('test.conll')
    genDictionaries(trainSents, trainTrees)

    embedding_filename = 'word2vec.model'

    embedding_array = load_embeddings(embedding_filename, wordDict, posDict, labelDict)

    labelInfo = []
    for idx in np.argsort(list(labelDict.values())):
        labelInfo.append(list(labelDict.keys())[idx])
    parsing_system = ParsingSystem(labelInfo[1:])
    print(parsing_system.rootLabel)

    print("Generating Traning Examples")
    trainFeats, trainLabels = genTrainExamples(trainSents, trainTrees)
    print("Done.")

    # Build the graph model
    graph = tf.Graph()
    model = DependencyParserModel(graph, embedding_array, Config)

    num_steps = Config.max_iter
    with tf.Session(graph=graph) as sess:

        model.train(sess, num_steps)

        model.evaluate(sess, testSents)
