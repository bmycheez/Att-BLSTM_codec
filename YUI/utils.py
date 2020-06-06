from random import shuffle, randint
from glob import glob
import time
import bitstring
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import math


def list2int(list):
    return int(len(list) != 0)


def primeSieve(sieveSize):
    # creating Sieve (0~n까지의 slot)
    sieve = [True] * (sieveSize+1)
    # 0과 1은 소수가 아니므로 제외
    sieve[0] = False
    sieve[1] = False
    # 2부터 (루트 n) + 1까지의 숫자를 탐색
    for i in range(2, int(math.sqrt(sieveSize))+1):
        # i가 소수가 아니면 pass
        if sieve[i] == False:
            continue
        # i가 소수라면 i*i~n까지 숫자 가운데 i의 배수를
        # 소수에서 제외
        for pointer in range(i**2, sieveSize+1, i):
            sieve[pointer] = False
    primes = []
    # sieve 리스트에서 True인 것이 소수이므로
    # True인 값의 인덱스를 결과로 저장
    for i in range(sieveSize+1):
        if sieve[i] == True:
            primes.append(i)
    return primes


codec_list = ['.m2v', '.h263', '.264', '.hevc', '.bit', '.webm', '.jpg', '.j2k', '.bmp', '.png', '.tiff']
codec = ['MPEG-2', 'H.263', 'H.264', 'H.265', 'IVC', 'VP8', 'JPEG', 'JPEG2000', 'BMP', 'PNG', 'TIFF']
alphabet = ['a', 'b', 'c', 'd', 'e', 'f']
scenario_list = ['default', 'inverse', 'xor']
# Bi-LSTM(Attention) Parameters
embedding_dim = 64  # 2
# n_hidden = 128       # 24
num_classes = len(codec_list)
# all_bytes_in_a_sentence = 32   # 128
shift_bytes_in_a_sentence = 1
num_chars_in_a_word = 1
dataset = 16       # 32
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
    def __init__(self, n_hidden):
        super(BiLSTM_Attention, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, n_hidden, bidirectional=True)
        self.out = nn.Linear(n_hidden * 2, num_classes)
        self.n_hidden = n_hidden

        # lstm_output : [batch_size, n_step, n_hidden * num_directions(=2)], F matrix

    def attention_net(self, lstm_output, final_state):
        hidden = final_state.view(-1, self.n_hidden * 2, 1)
        # hidden : [batch_size, n_hidden * num_directions(=2), 1(=n_layer)]
        # print(hidden[ind])
        attn_weights = torch.bmm(lstm_output, hidden).squeeze(2)
        # attn_weights : [batch_size, n_step]
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

        hidden_state = Variable(torch.zeros(1*2, len(X), self.n_hidden)).cuda()
        # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
        cell_state = Variable(torch.zeros(1*2, len(X), self.n_hidden)).cuda()
        # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]

        # final_hidden_state, final_cell_state : [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
        output, (final_hidden_state, final_cell_state) = self.lstm(input, (hidden_state, cell_state))
        output = output.permute(1, 0, 2) # output : [batch_size, len_seq, n_hidden]
        # print(output[ind])
        attn_output, attention = self.attention_net(output, final_hidden_state)
        return self.out(attn_output), attention # model : [batch_size, num_classes], attention : [batch_size, n_step]


def scenario_detect(frequency, video, count):
    if frequency.index(max(frequency)) == 0:                                                # MPEG2
        print(codec[frequency.index(max(frequency))])
        ssc = [hex2bin('000001b3')]                                                         # 스타트 코드들 저장
        sec = [hex2bin('000001b5')]
        gop = [hex2bin('000001b8')]
        psc = [hex2bin('00000100')]
    elif frequency.index(max(frequency)) == 1:                                              # H.263
        print(codec[frequency.index(max(frequency))])
        ssc = [hex2bin('000080'), hex2bin('000081'), hex2bin('000082'), hex2bin('000083')]  # psc
        # sec = [hex2bin('0000fc'), hex2bin('0000fd'), hex2bin('0000fe'), hex2bin('0000ff')]  # eos
        # gop = [hex2bin('0000f8'), hex2bin('0000f9')]                                        # eosbs
        # sec = []
        sec = []                                                                            # gbsc
        # """
        for i in range(8, 16):
            if i >= 11:
                j = alphabet[i - 11]
            else:
                j = str(i)
            sec.append(hex2bin('0000' + j))
        # """
        gop = []
        psc = []
    elif frequency.index(max(frequency)) == 2:                                              # H.264
        print(codec[frequency.index(max(frequency))])
        ssc = [hex2bin('0000000167')]                                                       # sps
        sec = [hex2bin('0000000168')]                                                       # pps
        gop = [hex2bin('0000000165'), hex2bin('0000010605')]                                # idr
        psc = [hex2bin('0000000141')]                                                       # nidr
    elif frequency.index(max(frequency)) == 3:                                              # H.265
        print(codec[frequency.index(max(frequency))])
        ssc = [hex2bin('000001')]                                                           # sps
        sec = [hex2bin('000003')]                                                           # pps
        gop = []                                                                            # idr
        psc = []                                                                            # nidr
    elif frequency.index(max(frequency)) == 4:                                              # IVC
        print(codec[frequency.index(max(frequency))])
        ssc = [hex2bin('000001b0')]                                                         # vsc
        sec = [hex2bin('00000100')]                                                         # vec
        gop = [hex2bin('000001b2')]                                                         # usc
        psc = [hex2bin('000001b3')]                                                         # udc
    elif frequency.index(max(frequency)) == 5:                                              # VP8
        print(codec[frequency.index(max(frequency))])
        ssc = [hex2bin('1a45dfa3010000000000001f')]                                         # sc
        sec = [hex2bin('7765626d')]                                                         # webm
        gop = [hex2bin('1549a96601')]                                                       # ed1
        psc = [hex2bin('00000000000032')]                                                   # ed2
    elif frequency.index(max(frequency)) == 6:                                              # JPEG
        print(codec[frequency.index(max(frequency))])
        ssc = [hex2bin('ffd8')]                                                             # sc
        sec = [hex2bin('ffc0'), hex2bin('ffc2')]                                            # sof
        gop = []                                                                            # None
        psc = []                                                                            # None
    elif frequency.index(max(frequency)) == 7:                                              # JPEG2000
        print(codec[frequency.index(max(frequency))])
        ssc = [hex2bin('ff4f')]                                                             # sc
        sec = [hex2bin('ff90')]                                                             # sot
        gop = [hex2bin('ff93')]                                                             # sod
        psc = []                                                                            # siz
    elif frequency.index(max(frequency)) == 8:                                              # BMP
        print(codec[frequency.index(max(frequency))])
        ssc = [hex2bin('424d')]                                                             # hd1
        sec = [hex2bin('28000000'), hex2bin('0c000000'), hex2bin('40000000'), hex2bin('6c000000'), hex2bin('7c000000')]
        gop = []                                                                            # V3
        psc = []                                                                            # None
    elif frequency.index(max(frequency)) == 9:                                              # PNG
        print(codec[frequency.index(max(frequency))])
        ssc = [hex2bin('89504e47')]                                                         # hd1
        sec = [hex2bin('49484452')]                                                         # ihdr
        gop = [hex2bin('49444154')]                                                         # idat
        psc = []                                                                            # None
    elif frequency.index(max(frequency)) == 10:                                             # TIFF
        print(codec[frequency.index(max(frequency))])
        ssc = [hex2bin('4949'), hex2bin('4d4d')]                                            # hd1
        sec = [hex2bin('002a'), hex2bin('2a00')]                                            # hd2
        gop = []                                                                            # None
        psc = []                                                                            # None
    all = list2int(ssc) + list2int(sec) + list2int(gop) + list2int(psc)
    hr = scenario_search(video, xor_header([ssc, sec, gop, psc], xor_flag=0))
    hx = scenario_search(video, xor_header([ssc, sec, gop, psc], xor_flag=1))
    video.pos = 0
    video = video.read(video.length).bin
    hh = [hr, hx]
    print(hh)
    condition = [True, True]
    for h in range(len(hh)):
        for l in range(all):
            if hh[h][l] == 0:
                condition[h] = False
            if codec[frequency.index(max(frequency))] == 'H.263' and hh[h][1] >= 10:
                condition[h] = False
        if condition[h]:
            detected_scenario = h + 1
            print('# of %s headers ->' % scenario_list[detected_scenario], hh[h])
            if detected_scenario == 1:
                video = encode(video, 'inv')
            elif detected_scenario == 2:
                video = dxor_fast(video, count)
            return detected_scenario, video
        else:
            continue
    print('Unknown scenario or codec mismatched!')
    print('Let me try the second best prediction!')
    frequency[frequency.index(max(frequency))] = 0
    video = bitstring.BitStream('0b' + video)
    detected_scenario, video = scenario_detect(frequency, video, count)
    return detected_scenario, video


def codec_decide(video, mode='image'):
    video.pos = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if mode == 'image':                 # 정보를 최대한 많이 받는 게 중요
        n_hidden = 32
        all_bytes_in_a_sentence = 128
    elif mode == 'video':               # 면밀한 분류를 하는 것이 중요
        n_hidden = 128
        all_bytes_in_a_sentence = 32
    model = BiLSTM_Attention(n_hidden)
    model.to(device)
    if mode == 'image':
        a = torch.load(glob('Bi-LSTM_96.73.pth')[0])
    elif mode == 'video':
        a = torch.load(glob('Bi-LSTM_97.94.pth')[0])
    model.load_state_dict(a)
    frequency = [0] * num_classes       # MPEG-2, H.263, H.264,... 의 예측값의 빈도수를 각각 저장
    limit = (dataset - 1) * shift_bytes_in_a_sentence + all_bytes_in_a_sentence     # 학습시킨 데이터 길이만큼만 검증
    for i in range(limit):
        tt = video.read(all_bytes_in_a_sentence * int(math.log2(16))).hex
        video.pos -= all_bytes_in_a_sentence * int(math.log2(16))
        video.pos += shift_bytes_in_a_sentence * int(math.log2(16))
        predict = test(tt, test_scenario, num_chars_in_a_word, model)
        frequency[predict] += 1
    return frequency


def xor_header(header_list, xor_flag=1):                                # 원래 코덱의 스타트 코드 검색 리스트
    a = len(header_list)
    b = []
    header_list_ = []                                                   # 시나리오로 변형된 코드 검색 리스트
    for i in range(a):
        header_list_.append([])
        b.append(len(header_list[i]))
        for j in range(b[i]):
            if xor_flag == 0:
                header_list_[i].append(encode(header_list[i][j], 'inv'))
            elif xor_flag == 1:
                header_list_[i].append(xor_fast(header_list[i][j]))
                # 스타트 코드 다음 한 비트가 0이냐 1이냐에 따라서 두 가지 결과가 나오기에 모두 저장
                if header_list_[i][j][len(header_list_[i][j]) - 1] == '0':
                    new = header_list_[i][j][:len(header_list_[i][j]) - 1]
                    new += '1'
                    header_list_[i].append(new)
                elif header_list_[i][j][len(header_list_[i][j]) - 1] == '1':
                    new = header_list_[i][j][:len(header_list_[i][j]) - 1]
                    new += '0'
                    header_list_[i].append(new)
    return header_list_


def scenario_search(video, header_list):
    frequency_header = [0] * len(header_list)
    time_scale = int(math.pow(10, 2)) // 2
    # 1일때 모든 비트스트림을 다 본다. 시간이 너무 길게 걸리므로 이 값을 늘리면서 실험하는 것을 추천.
    limit = (len(video) - len(header_list[0][0])) // time_scale
    for video.pos in range(limit):
        for k in range(len(header_list)):
            if header_list[k] == []:
                continue
            code = video.read(len(header_list[k][0])).bin
            if code in header_list[k]:
                frequency_header[k] += 1
    return frequency_header


def factor(n):
    result = []
    original = n
    for i in primeSieve(n):
        count = 0
        while n % i == 0:
            count += 1
            n = int(n/i)
        if count != 0:
            result.append((i, count))   # (소수, 개수) 형태로 리스트에 추가됨
        if n == 1:
            break
    # print(result)
    count = 1
    ref = 2
    for b in range(ref):
        count *= int(pow(result[b][0], result[b][1]))
    if count == original:
        count /= int(pow(result[1][0], result[1][1]))
    return int(count)


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
            # print(j + 1)
            m = string[j * k:(j + 1) * k]
            result = ''
            for i in range(len(m) - 1):
                partition = str(int(m[i]) ^ int(m[i + 1]))
                result += partition
            results.append(result)
            keys.append(m[1])
        keys.reverse()
    return ''.join(results) + ''.join(keys)


def encode(string, operator):
    result = ''
    if operator == 'inv':
        for i in range(len(string)):
            if string[i] == '1':
                result += '0'
            if string[i] == '0':
                result += '1'
    return result


def decode_all(string, operator, part):
    results = []
    k = len(string)//part - 1
    for j in range(part):
        m = string[j * k:(j + 1) * k]
        key = string[len(string) - 1 - j]
        result = decode(m + key, operator)
        results.append(result)
        # print(j/part*100, '%')
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
    return encode_all(string, 'xor', part)


def dxor_fast(string, part=1):
    return decode_all(string, 'xor', part)


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
                    b = bin2hex(xor_fast(hex2bin(b)))
                for number in range(int(w[2] * dataset)):
                    number *= num_chars
                    end = number + int(num_words_per_sentence * num_chars)
                    en = b[number:end]
                    if mode == you:
                        codec.append(en)
                        label.append(label_test)
                    else:
                        codec.append(bin2hex(xor_fast(hex2bin(en))))
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
