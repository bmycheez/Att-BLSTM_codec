import torch.optim as optim
from test import *


def main():
    # Preprocessing
    training_data = preProcessing(all_bytes_in_a_sentence, shift_bytes_in_a_sentence, num_chars_in_a_word, dataset,
                                  training_scenario, 'training_set')
    validation_data = preProcessing(all_bytes_in_a_sentence, shift_bytes_in_a_sentence, num_chars_in_a_word, dataset,
                                    training_scenario, 'validation_set')
    sentences = training_data[0]
    labels = training_data[1]
    validation_sentences = validation_data[0]
    validation_labels = validation_data[1]
    for i in range(len(sentences)):
        sentences[i] = split1to10(sentences[i], num_chars_in_a_word)
    for i in range(len(validation_sentences)):
        validation_sentences[i] = split1to10(validation_sentences[i], num_chars_in_a_word)
    inputs = []
    for sen in sentences:
        inputs.append(np.asarray([word_dict[n] for n in sen.split()]))
    targets = []
    for out in labels:
        targets.append(out)
    inputs_val = []
    for sen in validation_sentences:
        inputs_val.append(np.asarray([word_dict[n] for n in sen.split()]))
    targets_val = []
    for out in validation_labels:
        targets_val.append(out)
    input_batch = Variable(torch.LongTensor(inputs)).cuda()
    target_batch = Variable(torch.LongTensor(targets)).cuda()

    # Training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BiLSTM_Attention()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    lr = 1e-2
    optimizer = optim.Adam(model.parameters(), lr=lr)
    start = time.time()
    plot_losses = []
    print_loss_total = 0
    plot_loss_total = 0
    print_every = 100
    plot_every = print_every//1
    n_iters = 2000
    decay = 2
    step = n_iters//2
    print(codec_list)
    print('embedding_dim:', embedding_dim)
    print('n_hidden:', n_hidden)
    print('all_bytes_in_a_sentence:', all_bytes_in_a_sentence)
    print('dataset:', dataset)
    print('lr:', lr)
    print('decay:', decay)
    print('step:', step)
    acc_max = 0
    for epoch in range(1, n_iters + 1):
        current_lr = lr * ((1 / decay) ** (epoch // step))
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        model.train()
        optimizer.zero_grad()
        output, attention = model(input_batch)
        loss = criterion(output, target_batch)
        print_loss_total += loss.item()
        plot_loss_total += loss.item()
        if epoch % print_every == 0:
            model.eval()
            c = testall(validation_sentences, validation_labels, training_scenario, num_chars_in_a_word, model)
            acc = 0
            for i in range(num_classes):
                acc += (c[i][i]) * 100 / len(validation_sentences)
            # '''
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.10f %.2f' % (timeSince(start, epoch / n_iters), epoch, epoch / n_iters * 100,
                                               print_loss_avg, acc))
            if acc >= acc_max:
                acc_max = acc
                torch.save(model.state_dict(),
                           './Bi-LSTM_' + str(round(acc, 2)) + '.pth')
        if epoch % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
        loss.backward()
        optimizer.step()
    # Test
    detect(model, test_scenario)


if __name__ == "__main__":
    main()
