# training and validation loss were calculated after every epoch
import torch
from torch.cuda import device
import torch.optim as optim
import torch
from model import *
from dataset import *
import torch.utils.data as utils


# saving and loading checkpoint mechanisms
def save_checkpoint(save_path, model, optimizer, val_loss):
    if save_path == None:
        return
    save_path = save_path
    state_dict = {'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'val_loss': val_loss}

    torch.save(state_dict, save_path)

    print(f'Model saved to ==> {save_path}')


def load_checkpoint(model, optimizer):
    save_path = f'siameseNet-batchnorm50.pt'
    state_dict = torch.load(save_path)
    model.load_state_dict(state_dict['model_state_dict'])
    optimizer.load_state_dict(state_dict['optimizer_state_dict'])
    val_loss = state_dict['val_loss']
    print(f'Model loaded from <== {save_path}')

    return val_loss


def count_parameters(model):
    temp = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'The model architecture:\n\n', model)
    print(f'\nThe model has {temp:,} trainable parameters')


# training and validation after every epoch
def train(model, train_loader, val_loader, num_epochs, criterion, save_name):
    best_val_loss = float("Inf")
    train_losses = []
    val_losses = []
    cur_step = 0
    for epoch in range(num_epochs):
        running_loss = 0.0
        model.train()
        print("Starting epoch " + str(epoch + 1))
        for img1, img2, labels in train_loader:
            # Forward
            img1 = img1.to(device)
            img2 = img2.to(device)
            labels = labels.to(device)
            outputs = model(img1, img2)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        val_running_loss = 0.0
        with torch.no_grad():
            model.eval()
            for img1, img2, labels in val_loader:
                img1 = img1.to(device)
                img2 = img2.to(device)
                labels = labels.to(device)
                outputs = model(img1, img2)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()
        avg_val_loss = val_running_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print('Epoch [{}/{}],Train Loss: {:.4f}, Valid Loss: {:.8f}'
              .format(epoch + 1, num_epochs, avg_train_loss, avg_val_loss))
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_checkpoint(save_name, model, optimizer, best_val_loss)

    print("Finished Training")
    return train_losses, val_losses


# evaluation metrics
def eval(model, test_loader):
    with torch.no_grad():
        model.eval()
        correct = 0
        print('Starting Iteration')
        count = 0
        for mainImg, imgSets, label in test_loader:
            mainImg = mainImg.to(device)
            predVal = 0
            pred = -1
            for i, testImg in enumerate(imgSets):
                testImg = testImg.to(device)
                output = model(mainImg, testImg)
                if output > predVal:
                    pred = i
                    predVal = output
            label = label.to(device)
            if pred == label:
                correct += 1
            count += 1
            if count % 20 == 0:
                print("Current Count is: {}".format(count))
                print('Accuracy on n way: {}'.format(correct / count))


if __name__ == "__main__":
    # choose a training dataset size and further divide it into train and validation set 80:20
    dataSize = 10000  # self-defined dataset size
    TRAIN_PCT = 0.8  # percentage of entire dataset for training
    train_size = int(dataSize * TRAIN_PCT)
    val_size = dataSize - train_size

    transformations = transforms.Compose(
        [transforms.ToTensor()])

    omniglotDataset = OmniglotDataset(categories, root_dir, dataSize, transformations)
    train_set, val_set = random_split(omniglotDataset, [train_size, val_size])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=1, num_workers=0, shuffle=True)

    # create the test set for final testing
    testSize = 5000
    numWay = 20
    test_set = NWayOneShotEvalSet(categories, root_dir, testSize, numWay, transformations)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, num_workers=0, shuffle=True)

    # creating the original network and counting the paramenters of different networks
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    siameseBaseLine = Net()
    siameseBaseLine = siameseBaseLine.to(device)
    count_parameters(siameseBaseLine)

    #Train Setting
    optimizer = optim.Adam(siameseBaseLine.parameters(), lr = 0.0006)
    num_epochs = 50
    criterion = nn.BCEWithLogitsLoss()
    save_path = 'siameseNet-batchnorm50.pt'

    train_losses, val_losses = train(siameseBaseLine, train_loader, val_loader, num_epochs, criterion, save_path)