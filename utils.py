from random import shuffle, randint
from glob import glob
import os
import time
import math
import bitstring
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt

# Network Setting
codec_list = ['.m2v', '.h263', '.264', '.mp4', '.bit', '.webm', '.jpg', '.j2k', '.bmp', '.png', '.tiff']
alphabet = ['a', 'b', 'c', 'd', 'e', 'f']
# Bi-LSTM(Attention) Parameters
embedding_dim = 128
n_hidden = 64
num_classes = len(codec_list)
all_bytes_in_a_sentence = 64
shift_bytes_in_a_sentence = 1
num_chars_in_a_word = 1
dataset = 16
training_scenario = 3
test_scenario = 2

# Word List for Att-BLSTM
word_dict = {}
hexList = []
# ind = 1
# print(sentences[ind], labels[ind])
for i in range(10):
    hexList.append(str(i))
for i in alphabet:
    hexList.append(i)
for i in range(16):
    word_dict[hexList[i]] = i
    for j in range(16):
        word_dict[hexList[i]+hexList[j]] = 16 * i + j
        for k in range(16):
            word_dict[hexList[i]+hexList[j]+hexList[k]] = (16 ** 2) * i + 16 * j + k
            for l in range(16):
                word_dict[hexList[i]+hexList[j]+hexList[k]+hexList[l]] = (16 ** 3) * i + (16 ** 2) * j + 16 * k + l
                # '''
# print(sentences[ind], labels[ind])
vocab_size = len(word_dict)


# Network
class BiLSTM_Attention(nn.Module):
    def __init__(self):
        super(BiLSTM_Attention, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, n_hidden, bidirectional=True)
        self.out = nn.Linear(n_hidden * 2, num_classes)

    # lstm_output : [batch_size, n_step, n_hidden * num_directions(=2)], F matrix

    def attention_net(self, lstm_output, final_state):
        hidden = final_state.view(-1, n_hidden * 2, 1)
        # hidden : [batch_size, n_hidden * num_directions(=2), 1(=n_layer)]
        # print(hidden[ind])
        attn_weights = torch.bmm(lstm_output, hidden).squeeze(2) # attn_weights : [batch_size, n_step]
        # print(attn_weights[ind])
        soft_attn_weights = F.softmax(attn_weights, 1)
        # print(soft_attn_weights[ind])
        # [batch_size, n_hidden * num_directions(=2), n_step] * [batch_size, n_step, 1]
        # = [batch_size, n_hidden * num_directions(=2), 1]
        context = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        # print(context[ind])
        return context, soft_attn_weights.data # context : [batch_size, n_hidden * num_directions(=2)]

    def forward(self, X):
        input = self.embedding(X).cuda() # input : [batch_size, len_seq, embedding_dim]
        # print(input[ind])
        input = input.permute(1, 0, 2) # input : [len_seq, batch_size, embedding_dim]

        hidden_state = Variable(torch.zeros(1*2, len(X), n_hidden)).cuda()
        # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
        cell_state = Variable(torch.zeros(1*2, len(X), n_hidden)).cuda()
        # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]

        # final_hidden_state, final_cell_state : [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
        output, (final_hidden_state, final_cell_state) = self.lstm(input, (hidden_state, cell_state))
        output = output.permute(1, 0, 2) # output : [batch_size, len_seq, n_hidden]
        # print(output[ind])
        attn_output, attention = self.attention_net(output, final_hidden_state)
        return self.out(attn_output), attention # model : [batch_size, num_classes], attention : [batch_size, n_step]


def endian_swap_all(string, byte):
    result = ''
    for i in range(len(string)//byte):
        partition = string[i*byte:i*byte+byte]
        result += endian_swap(partition)
    return result


def endian_swap(string):
    string = string[::-1]
    result = ''
    for i in range(len(string)//2):
        partition = string[i*2:i*2+2]
        result += partition[::-1]
    return result


def encode_all(string, operator, part):
    results = []
    keys = []
    if operator == 'xor':
        k = len(string)//part
        for j in range(part):
            m = string[j * k:(j + 1) * k]
            result = ''
            for i in range(len(m) - 1):
                partition = str(int(m[i]) ^ int(m[i + 1]))
                result += partition
            results.append(result)
            keys.append(m[1])
        keys.reverse()
    return ''.join(results) + ''.join(keys)


def decode_all(string, operator, part):
    results = []
    k = len(string)//part - 1
    for j in range(part):
        m = string[j * k:(j + 1) * k]
        key = string[len(string) - 1 - j]
        result = decode(m + key, operator)
        results.append(result)
    return ''.join(results)


def decode(string, operator):
    result = []
    if operator == 'xor':
        for i in range(len(string) - 1):
            if i == 0:
                if string[i] == '0' and string[len(string) - 1] == '0':
                    result.append('00')
                if string[i] == '0' and string[len(string) - 1] == '1':
                    result.append('11')
                if string[i] == '1' and string[len(string) - 1] == '0':
                    result.append('10')
                if string[i] == '1' and string[len(string) - 1] == '1':
                    result.append('01')
            else:
                limit = len(result)
                for j in range(limit):
                    if string[i] == '0' and result[j][i] == '0':
                        result.append(result[j] + '0')
                    elif string[i] == '0' and result[j][i] == '1':
                        result.append(result[j] + '1')
                    elif string[i] == '1' and result[j][i] == '1':
                        result.append(result[j] + '0')
                    elif string[i] == '1' and result[j][i] == '0':
                        result.append(result[j] + '1')
                result.reverse()
                for k in range(limit):
                    result.pop()
    return result[0]


def xor_fast(string, part=1):
    return bin2hex(encode_all(hex2bin(string), 'xor', part))


def dxor_fast(string, part=1):
    return bin2hex(decode_all(hex2bin(string), 'xor', part))


def dec2bin(number, length):
    result = ''
    if number == 0:
        return '0000'
    while number != 1:
        result += str(number%2)
        number = number//2
    result += '1'
    result = result[::-1]
    final = ''
    for i in range(length):
        if len(result) == length - i:
            for j in range(i):
                final += '0'
            final += result
    return final


def bin2dec(string):
    dec = 0
    for n in range(len(string)):
        dec += int(string[n]) * pow(2, len(string) - n - 1)
    return dec


def hex2bin(string):
    global alphabet
    result = ''
    hex2dec = list(range(10, 16))
    for i in range(len(string)):
        for j in range(10):
            if string[i] == str(j):
                result += dec2bin(j, 4)
        for j in range(6):
            if string[i] == alphabet[j]:
                result += dec2bin(hex2dec[j], 4)
    return result


def bin2hex(string):
    global alphabet
    result = ''
    for i in list(range(0, len(string), 4)):
        s = string[i:i+4]
        t = bin2dec(s)
        for j in range(10):
            if t == j:
                result += str(j)
        for j in range(6):
            if t == j + 10:
                result += alphabet[j]
    return result


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def Hex2Zero(string):
    global alphabet
    result = ''
    # Scenario : Reversed 0 & 1 -> Call This Function before calling SplitOne2Ten
    for i in range(len(string)):
        for j in range(len(alphabet)):
            if string[i] == alphabet[j]:
                inputnumber = 10 + j
        if string[i] not in alphabet:
            inputnumber = int(string[i])
        outputnumber = 15 - inputnumber
        for j in range(len(alphabet)):
            if outputnumber == 10 + j:
                part = alphabet[j]
        if outputnumber < 10:
            part = str(outputnumber)
        result += part
    return result


def split1to10(string, word_length):        # 1-byte N words
    original = string
    index = word_length
    sentence_length = len(string) // word_length
    string = original[ : index]
    for i in range(sentence_length - 1):
        string = string + ' '
        string = string + original[index : (index + word_length)]
        index += word_length
    string += original[index : ]
    return string


def preProcessing(num_words_per_sentence, shift, num_chars, dataset, training_scenario, mode):
    global codec_list
    codec = []
    label = []
    you = 'test_set'
    w = [1, 1, 1, 1]
    if mode == you:
        codec_list2 = []
        label_test = randint(0, len(codec_list) - 1)
        codec_list2.append(codec_list[label_test])
    else:
        codec_list2 = codec_list
    for i in range(len(codec_list2)):
        # print(i)
        files = glob('D:/' + mode + '/*' + codec_list2[i])
        # print(files)
        if mode == you:
            files2 = []
            file_test = randint(0, len(files) - 1)
            files2.append(files[file_test])
        else:
            files2 = files
        for j in range(len(files2)):
            b = bitstring.ConstBitArray(filename=files2[j]).hex
            original = b
            print(files2[j])
            if (mode == you and training_scenario in [0]) or \
                    (mode != you and training_scenario in [0, 1, 2, 3]):
                for number in range(int(w[0] * dataset)):
                    number *= num_chars
                    end = number + int(num_words_per_sentence * num_chars)
                    en = b[number:end]
                    codec.append(en)
                    if mode == you:
                        label.append(label_test)
                    else:
                        label.append(i)
            if (mode == you and training_scenario in [1]) or \
                    (mode != you and training_scenario in [1, 3]):
                if mode == you:
                    b = Hex2Zero(b)
                for number in range(int(w[1] * dataset)):
                    number *= num_chars
                    end = number + int(num_words_per_sentence * num_chars)
                    en = b[number:end]
                    if mode == you:
                        codec.append(en)
                        label.append(label_test)
                    else:
                        codec.append(Hex2Zero(en))
                        label.append(i)
            if (mode == you and training_scenario in [2]) or \
                    (mode != you and training_scenario in [2, 3]):
                if mode == you:
                    b = xor_fast(b)
                for number in range(int(w[2] * dataset)):
                    number *= num_chars
                    end = number + int(num_words_per_sentence * num_chars)
                    en = b[number:end]
                    if mode == you:
                        codec.append(en)
                        label.append(label_test)
                    else:
                        codec.append(xor_fast(en))
                        label.append(i)
            # """
            if (mode == you and training_scenario in [4]) or \
                    (mode != you and training_scenario in [4, 3]):
                if mode == you:
                    b = endian_swap_all(b, 4)
                for number in range(int(w[3] * dataset)):
                    number *= num_chars
                    end = number + int(num_words_per_sentence * num_chars)
                    en = b[number:end]
                    if mode == you:
                        codec.append(en)
                        label.append(label_test)
                    else:
                        codec.append(endian_swap(en))
                        label.append(i)
            # """
    if mode == 'training_set':
        result = shufflemylist(codec, label)
    else:
        result = []
        result.append(codec)
        result.append(label)
    if mode == you:
        result.append(original)
        result.append(b)
    return result


def shufflemylist(random_codec, random_label):
    order = list(range(len(random_codec)))
    shuffle(order)
    final_codec = []
    final_label = []
    for i in range(len(order)):
        final_codec.append(random_codec[order[i]])
        final_label.append(random_label[order[i]])
    result = []
    result.append(final_codec)
    result.append(final_label)
    return result


def test(test_text, scenario, num_chars_in_a_word, model):
    # Test
    test_text = test_text.replace(" ", "")
    test_text = split1to10(test_text, num_chars_in_a_word)
    tests = [np.asarray([word_dict[n] for n in test_text.split()])]
    test_batch = Variable(torch.LongTensor(tests)).cuda()
    # Predict
    predict, _ = model(test_batch)
    predict = predict.data.max(1, keepdim=True)[1]
    return predict[0][0]


def testall(test_sentences, test_labels, scenario, num_chars_in_a_word, model):
    global num_classes
    confusion_matrix = np.zeros((num_classes, num_classes))
    for i in range(len(test_sentences)):
        predict = test(test_sentences[i], scenario, num_chars_in_a_word, model)
        for j in range(num_classes):
            for k in range(num_classes):
                if predict == j and test_labels[i] == k:
                    confusion_matrix[j][k] += 1
    return confusion_matrix


def show_matrix(c):
    print(c)
    fig = plt.figure()
    # [predict][true]
    ax = fig.add_subplot(1, 1, 1)
    cax = ax.matshow(c, cmap='BuPu')
    fig.colorbar(cax)
    cm_label = ['2', '3', '8', 'J', 'B', 'T']
    ax.set_xticklabels(['']+cm_label, fontdict={'fontsize': 14})
    ax.set_yticklabels(['']+cm_label, fontdict={'fontsize': 14})
    plt.show()
    return