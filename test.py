from utils import *


def detect(model=0, test_scenario=2):
    trial = 10
    if model == 0:
        # """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = BiLSTM_Attention()
        model.to(device)
        # """
        a = torch.load(glob('./Bi-LSTM_96.09.pth')[0])
        model.load_state_dict(a)
        # """
    else:
        torch.save(model.state_dict(),
                   './Bi-LSTM_.pth')
    model.eval()
    acc_all = 0
    for j in range(trial):
        print(j, end=' ')
        test_data = preProcessing(all_bytes_in_a_sentence, shift_bytes_in_a_sentence, num_chars_in_a_word, dataset,
                                  test_scenario, 'test_set')
        test_sentences = test_data[0]
        test_labels = test_data[1]
        for i in range(len(test_sentences)):
            test_sentences[i] = split1to10(test_sentences[i], num_chars_in_a_word)
        # Detect
        c = testall(test_sentences, test_labels, test_scenario, num_chars_in_a_word, model)
        acc = 0
        for k in range(num_classes):
            acc += (c[k][k]) * 100 / len(test_sentences)
        print('Accuracy = %.2f%%' % acc)
        # show_matrix(c)
        acc_all += acc
    acc_all /= trial
    print('Accuracy = ' + str(round(acc_all, 2)) + '%')


if __name__ == "__main__":
    detect()
