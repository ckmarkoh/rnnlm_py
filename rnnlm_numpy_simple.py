from collections import Counter, OrderedDict
import math
import numpy as np
import copy
import json
import sys


class CRnnLM(object):
    def __init__(self, params):
        self.line_end = "</s>"
        self.train_file = params.get("train", "train")
        self.valid_file = params.get("valid", "valid")
        self.vocab = []
        self.vocab_hash = {}
        self.train_words = -1
        self.vocab_size = -1
        self.layer0_size = -1
        self.layer1_size = params.get("hidden", 30)
        self.layer2_size = -1
        self.class_size = params.get("class", 100)
        self.neu0 = None
        self.neu1 = None
        self.neu2 = None
        self.syn0 = None
        self.syn1 = None
        self.neu0b = None
        self.neu1b = None
        self.neu2b = None
        self.syn0b = None
        self.syn1b = None
        self.bptt = params.get("bptt", 0)
        self.bptt_history = 0
        self.bptt_block = params.get("bptt-block", 10)
        self.bptt_hidden = None
        self.bptt_syn0 = None
        self.class_word = None
        self.class_cn = None
        self.counter = 0
        self.alpha = 0.1
        self.beta = 0.0000001
        self.logp = 0
        self.llogp = -100000000
        self.min_improvement = 1.003
        self.alpha_divide = 0
        self.model_name = params.get("rnnlm", "model.json")

    def gen_neuron(self, width):
        return {
            "ac": np.zeros((1, width)),
            "er": np.zeros((1, width)),
        }

    def gen_vocab(self, w, c=0):
        return {
            "word": w,
            "cn": c,
            "prob": 0.0,
            "class_index": -1,
        }

    def gen_weight(self, shape):
        return 0.2 * (0.5 - np.random.rand(*shape)) + \
               0.2 * (0.5 - np.random.rand(*shape)) + \
               0.2 * (0.5 - np.random.rand(*shape))

    def read_file(self, fname):
        f = open(fname, "r")
        for line in f.readlines():
            words = line.replace("\n", " " + self.line_end).split(" ")
            for word in words:
                if len(word) > 0:
                    yield word

    def clip_val(self, mat, val=50):
        mat[mat > val] = val
        mat[mat < -val] = -val

    def sigmoid(self, mat):
        mat[:] = 1 / (1 + np.exp(-mat))

    def softmax(self, mat):
        mat_max = np.max(mat)
        mat_exp = np.exp(mat - mat_max)
        mat[:] = mat_exp / np.sum(mat_exp)

    def learnVocabFromTrainFile(self):
        count_words = Counter()
        # for line in f.readlines():
        #    words = line.replace("\n", " </s>").split(" ")
        count_words.update([self.line_end])
        count_words.update(self.read_file(self.train_file))
        self.vocab_size = len(count_words)
        self.train_words = sum(count_words.values())
        for wc in count_words.most_common():
            # if len(wc[0]) > 0:
            w = wc[0]
            c = wc[1]
            self.vocab.append(self.gen_vocab(w, c))
            self.vocab_hash.update({w: len(self.vocab_hash)})

    def initNet(self):
        self.layer0_size = self.vocab_size + self.layer1_size
        self.layer2_size = self.vocab_size + self.class_size

        self.neu0 = self.gen_neuron(self.layer0_size)
        self.neu1 = self.gen_neuron(self.layer1_size)
        self.neu2 = self.gen_neuron(self.layer2_size)

        self.neu0b = copy.deepcopy(self.neu0)
        self.neu1b = copy.deepcopy(self.neu1)
        self.neu2b = copy.deepcopy(self.neu2)

        self.syn0 = self.gen_weight((self.layer0_size, self.layer1_size))
        self.syn1 = self.gen_weight((self.layer1_size, self.layer2_size))

        self.syn0b = copy.deepcopy(self.syn0)
        self.syn1b = copy.deepcopy(self.syn1)

        self.bptt_history = np.zeros((self.bptt + self.bptt_block + 10,)).astype(int)
        self.bptt_history[:self.bptt + self.bptt_block] = -1
        self.bptt_hidden = self.gen_neuron((self.bptt + self.bptt_block + 1) * self.layer1_size)
        self.bptt_syn0 = self.gen_weight((self.layer0_size, self.layer1_size))
        self.bptt_syn0[:] = 0

        dd = sum([math.sqrt(vocab["cn"]) / float(self.train_words) for vocab in self.vocab])
        df = 0
        a = 0
        for vocab in self.vocab:
            df += (math.sqrt(vocab["cn"]) / float(self.train_words)) / float(dd)
            if df > 1:
                df = 1
            vocab["class_index"] = a
            if df > ((a + 1) / float(self.class_size)):
                if a < self.class_size - 1:
                    a += 1
        self.class_word = [[] for i in range(self.class_size)]
        self.class_cn = np.zeros(self.class_size).astype(int)
        for i, vocab in enumerate(self.vocab):
            cl = vocab["class_index"]
            self.class_word[cl].append(i)
            self.class_cn[cl] += 1

    def loadNet(self, fname):
        f = open(fname)
        json_item = json.loads("".join(f.readlines()))
        f.close()
        self.vocab = json_item["vocab"]
        self.vocab_size = len(json_item["vocab"])
        self.train_words = json_item["train_words"]
        for vocab in self.vocab:
            self.vocab_hash.update({vocab["word"]: len(self.vocab_hash)})
        self.initNet()
        self.syn0[:] = np.array(json_item["syn0"])[:]
        self.syn1[:] = np.array(json_item["syn1"])[:]
        self.alpha = json_item["alpha"]
        self.llogp = json_item["llogp"]

    def saveWeights(self):
        self.neu0b["ac"][:] = self.neu0["ac"][:]
        self.neu1b["ac"][:] = self.neu1["ac"][:]
        self.neu2b["ac"][:] = self.neu2["ac"][:]
        self.neu0b["er"][:] = self.neu0["er"][:]
        self.neu1b["er"][:] = self.neu1["er"][:]
        self.neu2b["er"][:] = self.neu2["er"][:]

        self.syn0b[:] = self.syn0[:]
        self.syn1b[:] = self.syn1[:]

    def restoreWeights(self):
        self.neu0["ac"][:] = self.neu0b["ac"][:]
        self.neu1["ac"][:] = self.neu1b["ac"][:]
        self.neu2["ac"][:] = self.neu2b["ac"][:]
        self.neu0["er"][:] = self.neu0b["er"][:]
        self.neu1["er"][:] = self.neu1b["er"][:]
        self.neu2["er"][:] = self.neu2b["er"][:]

        self.syn0[:] = self.syn0b[:]
        self.syn1[:] = self.syn1b[:]

    def matrixXvector(self, dest, srcvec, srcmatrix, mfrom, mto, mfrom2, mto2, type):
        if type == 0:
            dest[0, mfrom:mto] += np.dot(srcvec[0, mfrom2:mto2], srcmatrix[mfrom2:mto2, mfrom:mto])
        else:
            dest[0, mfrom2:mto2] += np.dot(srcvec[0, mfrom:mto], srcmatrix.T[mfrom:mto, mfrom2:mto2])

    def computeNet(self, last_word, word):
        if last_word != -1:
            self.neu0["ac"][0, last_word] = 1
        self.neu1["ac"][:] = 0
        self.neu2["ac"][:] = 0
        self.matrixXvector(self.neu1["ac"], self.neu0["ac"], self.syn0, 0, self.layer1_size,
                           self.layer0_size - self.layer1_size,
                           self.layer0_size, 0)
        self.matrixXvector(self.neu1["ac"], self.neu0["ac"], self.syn0, 0, self.layer1_size, last_word, last_word + 1,
                           0)
        self.clip_val(self.neu1["ac"], 50)
        self.sigmoid(self.neu1["ac"])
        self.matrixXvector(self.neu2["ac"], self.neu1["ac"], self.syn1, self.vocab_size, self.layer2_size,
                           0, self.layer1_size, 0)
        self.softmax(self.neu2["ac"][0, self.vocab_size:self.layer2_size])
        word_class_idx = self.vocab[word]["class_index"]
        word_pos_1 = self.class_word[word_class_idx][0]
        word_pos_2 = word_pos_1 + self.class_cn[word_class_idx]
        self.matrixXvector(self.neu2["ac"], self.neu1["ac"], self.syn1, word_pos_1, word_pos_2, 0, self.layer1_size, 0)
        self.softmax(self.neu2["ac"][0, word_pos_1:word_pos_2])

    def learnNet(self, last_word, word):
        beta2 = self.beta * self.alpha
        word_class_idx = self.vocab[word]["class_index"]
        word_pos_1 = self.class_word[word_class_idx][0]
        word_pos_2 = word_pos_1 + self.class_cn[word_class_idx]
        self.neu2["er"][0, word_pos_1:word_pos_2] = (0 - self.neu2["ac"][0, word_pos_1:word_pos_2])
        self.neu2["er"][0, word] = (1 - self.neu2["ac"][0, word])
        self.neu2["er"][0, self.vocab_size:self.layer2_size] = (
            0 - self.neu2["ac"][0, self.vocab_size:self.layer2_size])
        self.neu2["er"][0, self.vocab[word]["class_index"] + self.vocab_size] = (
            1 - self.neu2["ac"][0, self.vocab[word]["class_index"] + self.vocab_size])
        self.neu1["er"][:] = 0

        self.matrixXvector(self.neu1["er"], self.neu2["er"], self.syn1, word_pos_1, word_pos_2, 0, self.layer1_size, 1)

        if self.counter % 10 == 0:
            self.syn1[:, word_pos_1:word_pos_2] += \
                self.alpha * np.dot(self.neu1["ac"].T, self.neu2["er"][:, word_pos_1:word_pos_2]) \
                - self.syn1[:, word_pos_1:word_pos_2] * beta2
        else:
            self.syn1[:, word_pos_1:word_pos_2] += \
                self.alpha * np.dot(self.neu1["ac"].T, self.neu2["er"][:, word_pos_1:word_pos_2])

        self.matrixXvector(self.neu1["er"], self.neu2["er"], self.syn1, self.vocab_size, self.layer2_size, 0,
                           self.layer1_size, 1)

        if self.counter % 10 == 0:
            self.syn1[:, self.vocab_size:self.layer2_size] += \
                self.alpha * np.dot(self.neu1["ac"].T, self.neu2["er"][:, self.vocab_size:self.layer2_size]) \
                - self.syn1[:, self.vocab_size:self.layer2_size] * beta2
        else:
            self.syn1[:, self.vocab_size:self.layer2_size] += \
                self.alpha * np.dot(self.neu1["ac"].T, self.neu2["er"][:, self.vocab_size:self.layer2_size])

        if self.bptt <= 1:
            self.neu1["er"][0, :] = self.neu1["er"][0, :] * self.neu1["ac"][0, :] * (
                1 - self.neu1["ac"][:])  # error derivation at layer 1
            a = last_word
            if a != -1:
                if self.counter % 10 == 0:
                    self.syn0[a:a + 1, :] += self.alpha * np.dot(self.neu0["ac"][:, a:a + 1].T, self.neu1["er"][:]) \
                                             - self.syn0[a:a + 1, :] * beta2
                else:
                    self.syn0[a:a + 1, :] += self.alpha * np.dot(self.neu0["ac"][:, a:a + 1].T, self.neu1["er"][:])
            if self.counter % 10 == 0:
                self.syn0[self.vocab_size:self.layer0_size, :] += \
                    self.alpha * np.dot(self.neu0["ac"][:, self.vocab_size:self.layer0_size].T, self.neu1["er"][:]) \
                    - self.syn0[self.vocab_size:self.layer0_size, :] * beta2
            else:
                self.syn0[self.vocab_size:self.layer0_size, :] += \
                    self.alpha * np.dot(self.neu0["ac"][:, self.vocab_size:self.layer0_size].T, self.neu1["er"][:]) \
                    - self.syn0[self.vocab_size:self.layer0_size, :] * beta2
        else:
            self.bptt_hidden["ac"][0, :self.layer1_size] = self.neu1["ac"]
            self.bptt_hidden["er"][0, :self.layer1_size] = self.neu1["er"]

            if self.counter % self.bptt_block == 0:
                for step in range(self.bptt + self.bptt_block - 2):
                    self.neu1["er"] *= (self.neu1["ac"] * (1 - self.neu1["ac"]))
                    a = self.bptt_history[step]
                    if a != -1:
                        self.bptt_syn0[a:a + 1, :] += self.alpha * np.dot(np.ones((1,)), self.neu1["er"])
                    self.neu0["er"][0, self.vocab_size:self.layer0_size] = 0
                    self.matrixXvector(self.neu0["er"], self.neu1["er"], self.syn0, 0, self.layer1_size,
                                       self.vocab_size, self.layer0_size, 1)
                    self.bptt_syn0[self.vocab_size:self.layer0_size, :] += \
                        self.alpha * np.dot(self.neu0["ac"][:, self.vocab_size:self.layer0_size].T, self.neu1["er"])
                    self.neu1["er"][0, :] = self.neu0["er"][0, self.vocab_size:self.layer0_size] + \
                                            self.bptt_hidden["er"][0,
                                            (step + 1) * self.layer1_size:(step + 2) * self.layer1_size]
                    if step < self.bptt + self.bptt_block - 3:
                        self.neu1["ac"][0, :] = \
                            self.bptt_hidden["ac"][0, (step + 1) * self.layer1_size:(step + 2) * self.layer1_size]
                        self.neu0["ac"][0, self.vocab_size:self.layer0_size] = \
                            self.bptt_hidden["ac"][0, (step + 2) * self.layer1_size:(step + 3) * self.layer1_size]
                self.bptt_hidden["er"][0, :(self.bptt + self.bptt_block) * self.layer1_size] = 0
                self.neu1["ac"][:] = self.bptt_hidden["ac"][0, :self.layer1_size]

                if self.counter % 10 == 0:
                    self.syn0[self.vocab_size:self.layer0_size, :] += \
                        self.bptt_syn0[self.vocab_size:self.layer0_size, :] - \
                        self.syn0[self.vocab_size:self.layer0_size, :] * beta2
                else:
                    self.syn0[self.vocab_size:self.layer0_size, :] += \
                        self.bptt_syn0[self.vocab_size:self.layer0_size, :]
                self.bptt_syn0[self.vocab_size:self.layer0_size, :] = 0
                for step in range(self.bptt + self.bptt_block - 2):
                    a = self.bptt_history[step]
                    if a != -1:
                        if self.counter % 10 == 0:
                            self.syn0[a:a + 1, :] += self.bptt_syn0[a:a + 1, :] - self.syn0[a:a + 1, :] * beta2
                            self.bptt_syn0[a:a + 1, :] = 0
                        else:
                            self.syn0[a:a + 1, :] += self.bptt_syn0[a:a + 1, :]
                            self.bptt_syn0[a:a + 1, :] = 0

    def copyHiddenLayerToInput(self):
        self.neu0["ac"][0, self.vocab_size:self.layer0_size] = self.neu1["ac"][0, :]

    def netFlush(self):
        self.neu1["ac"][:] = 0
        self.neu2["ac"][:] = 0
        self.neu1["er"][:] = 0
        self.neu2["er"][:] = 0

    def readWordIndex(self, fi):
        for word_str in self.read_file(fi):
            word = self.vocab_hash.get(word_str, -1)
            yield word

    def saveNet(self, fname):
        f = open(fname, "w")
        json_item = {
            "train_words": self.train_words,
            "vocab": self.vocab,
            "syn0": self.syn0.tolist(),
            "syn1": self.syn1.tolist(),
            "alpha": self.alpha,
            "llogp": self.llogp,
        }
        f.write(json.dumps(json_item, indent=4))
        f.close()

    def trainNet(self, model=""):
        if len(model) > 0:
            self.loadNet(model)
        else:
            self.learnVocabFromTrainFile()
            self.initNet()
        for it in range(10000):
            last_word = 0
            self.counter = 0
            self.logp = 0

            if self.bptt > 0:
                self.bptt_history[:self.bptt + self.bptt_block] = 0

            for word in self.readWordIndex(self.train_file):
                self.counter += 1
                self.computeNet(last_word=last_word, word=word)
                if self.counter % 10000 == 0:
                    print "Iter: %3d\tAlpha: %f\t TRAIN entropy: %.4f  Progress: %.2f " \
                          % (it, self.alpha, (-self.logp / math.log(2, 10)) / float(self.counter),
                             (self.counter / float(self.train_words)) * 100)
                if word != -1:
                    self.logp += math.log(
                        self.neu2["ac"][0, self.vocab[word]["class_index"] + self.vocab_size] * self.neu2["ac"][
                            0, word], 10)
                if self.bptt > 0:
                    self.bptt_history[:self.bptt + self.bptt_block] = \
                        np.roll(self.bptt_history[:self.bptt + self.bptt_block], 1)
                    self.bptt_history[0] = last_word
                    self.bptt_hidden["ac"][0, :] = np.roll(self.bptt_hidden["ac"][0, :], self.layer1_size)
                    self.bptt_hidden["er"][0, :] = np.roll(self.bptt_hidden["er"][0, :], self.layer1_size)
                self.learnNet(last_word=last_word, word=word)
                self.copyHiddenLayerToInput()
                if last_word != -1:
                    self.neu0["ac"][0, last_word] = 0
                last_word = word
            last_word = 0
            self.logp = 0
            wordcn = 0
            for word in self.readWordIndex(self.valid_file):
                self.computeNet(last_word=last_word, word=word)
                if self.counter % 10000 == 0:
                    print "Iter: %3d\tAlpha: %f\t TRAIN entropy: %.4f  Progress: %.2f " \
                          % (it, self.alpha, (-self.logp / math.log(2, 10)) / float(self.counter),
                             (self.counter / float(self.train_words)) * 100)
                if word != -1:
                    self.logp += math.log(
                        self.neu2["ac"][0, self.vocab[word]["class_index"] + self.vocab_size] * self.neu2["ac"][
                            0, word],
                        10)
                    wordcn += 1
                self.copyHiddenLayerToInput()
                if last_word != -1:
                    self.neu0["ac"][0, last_word] = 0
                last_word = word
            if self.logp < self.llogp:
                self.restoreWeights()
            else:
                self.saveWeights()

            if self.logp * self.min_improvement < self.llogp:
                if self.alpha_divide == 0:
                    self.alpha_divide = 1
                else:
                    self.saveNet(self.model_name)
                    break

            print "VALID entropy: %.4f" % (-self.logp / math.log(2, 10) / float(wordcn))
            if self.alpha_divide:
                self.alpha /= 2

            self.llogp = self.logp
            self.logp = 0

            self.saveNet(self.model_name)


def main():
    model = CRnnLM({
        "train": "train",
        "valid": "valid",
        "bptt": 2,
        "bptt-block": 10,
    })
    model.trainNet()


if __name__ == "__main__":
    main()
