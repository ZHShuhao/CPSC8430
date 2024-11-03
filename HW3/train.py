import os
import torch
from net_model import QANet
from dataset import SpokenSQuADDataset
from tqdm import tqdm
import matplotlib.pyplot as plt


def main():
    epochs = 50
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = QANet()
    model.to(device)

    train_data_path = './Spoken-SQuAD/spoken_train-v1.1.json'
    test_data_path = './Spoken-SQuAD/spoken_test-v1.1.json'
    train_dataset = SpokenSQuADDataset(train_data_path, phase='train')
    test_dataset = SpokenSQuADDataset(test_data_path, phase='train')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=16)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=16)

    # model.bert.requires_grad = False
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    train_losses = []
    test_losses = []

    for epoch in range(epochs):
        progress_bar = tqdm(train_loader)
        epoch_loss = 0
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            input_mask = batch['input_mask'].to(device)
            start_pos = batch['start_pos'].to(device)
            end_pos = batch['end_pos'].to(device)
            optimizer.zero_grad()
            # print(input_ids.shape, input_mask.shape)
            start_logits, end_logits = model(input_ids, input_mask)
            loss = criterion(start_logits, start_pos) + criterion(end_logits, end_pos)
            loss.backward()
            optimizer.step()
            progress_bar.set_description(f'Epoch: {epoch}/{epochs}, loss: {loss.item()}')
            epoch_loss += loss.item()
        epoch_loss /= len(train_loader)
        scheduler.step(epoch_loss)
        train_losses.append(epoch_loss)

        # evaluate the model
        model.eval()
        with torch.no_grad():
            test_loss = 0
            for batch in test_loader:
                input_ids = batch['input_ids'].to(device)
                input_mask = batch['input_mask'].to(device)
                start_pos = batch['start_pos'].to(device)
                end_pos = batch['end_pos'].to(device)
                start_logits, end_logits = model(input_ids, input_mask)
                loss = criterion(start_logits, start_pos) + criterion(end_logits, end_pos)
                test_loss += loss.item()
            test_loss /= len(test_loader)
            print(f'Epoch: {epoch}/{epochs}, test_loss: {test_loss}')
            test_losses.append(test_loss)

        torch.save(model.state_dict(), f'qanet_{epoch}.pt')

    # plot the loss curve
    fig = plt.figure()
    plt.plot(train_losses, label='train_loss')
    plt.plot(test_losses, label='test_loss')
    plt.legend()
    fig.savefig('loss_curve.png')

    # save the model
    torch.save(model.state_dict(), 'qanet_final.pt')


if __name__ == "__main__":
    main()