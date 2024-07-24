import os, cv2,itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
from PIL import Image
import torch
from torch import optim,nn
from torch.autograd import Variable
from torch.utils.data import DataLoader,Dataset
from torchvision import models,transforms
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

np.random.seed(10)
torch.manual_seed(10)
torch.cuda.manual_seed(10)

print(os.listdir("../Data"))
data_dir = '../Data/HAM10000'
all_image_path = glob(os.path.join(data_dir, '*', '*.jpg'))
imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x for x in all_image_path}
lesion_type_dict = {
    'nv': 'Melanocytic nevi',
    'mel': 'dermatofibroma',
    'bkl': 'Benign keratosis-like lesions ',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
}

def compute_img_mean_std(image_paths):
    img_h, img_w = 224, 224
    imgs = []
    means, stdevs = [], []

    for i in tqdm(range(len(image_paths))):
        img = cv2.imread(image_paths[i])
        img = cv2.resize(img, (img_h, img_w))
        imgs.append(img)

    imgs = np.stack(imgs, axis=3)
    print(imgs.shape)

    imgs = imgs.astype(np.float32) / 255.

    for i in range(3):
        pixels = imgs[:, :, i, :].ravel()  # resize to one row
        means.append(np.mean(pixels))
        stdevs.append(np.std(pixels))

    means.reverse()  # BGR --> RGB
    stdevs.reverse()

    print("normMean = {}".format(means))
    print("normStd = {}".format(stdevs))
    return means,stdevs
norm_mean,norm_std = compute_img_mean_std(all_image_path)
# HAM10000 mean std
# norm_mean = [0.7630331, 0.5456457, 0.5700467]
# norm_std = [0.1409281, 0.15261227, 0.16997086]

df_original = pd.read_csv(os.path.join(data_dir, 'HAM10000_metadata.csv'))
df_original['path'] = df_original['image_id'].map(imageid_path_dict.get)
df_original['cell_type'] = df_original['dx'].map(lesion_type_dict.get)
df_original['cell_type_idx'] = pd.Categorical(df_original['cell_type']).codes
df_original.head()
_, df_val = train_test_split(df_original, test_size=0.2, random_state=101, stratify=df_original['cell_type_idx'])

df_val['cell_type_idx'].value_counts()

def get_val_rows(x):
    val_list = list(df_val['image_id'])
    if str(x) in val_list:
        return 'val'
    else:
        return 'train'

df_original['train_or_val'] = df_original['image_id']
df_original['train_or_val'] = df_original['train_or_val'].apply(get_val_rows)
df_train = df_original[df_original['train_or_val'] == 'train']
print(len(df_train))
print(len(df_val))

df_train['cell_type_idx'].value_counts()
df_val['cell_type'].value_counts()

df_train = df_train.reset_index()
df_val = df_val.reset_index()

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
class HAM10000(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # Load data and get label
        X = Image.open(self.df['path'][index])
        y = torch.tensor(int(self.df['cell_type_idx'][index]))

        if self.transform:
            X = self.transform(X)

        return X, y
def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    model_ft = None
    input_size = 0
    if model_name == "resnet":
        model_ft = models.resnet50()
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    else:
        print("Invalid model name, exiting...")
        exit()
    return model_ft, input_size
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def main():
    model_name = 'resnet'
    num_classes = 7
    feature_extract = False
    model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
    device = torch.device('cuda:0')
    model = model_ft.to(device)

    train_transform = transforms.Compose([transforms.Resize((input_size,input_size)),transforms.RandomHorizontalFlip(),
                                          transforms.RandomVerticalFlip(),transforms.RandomRotation(20),
                                          transforms.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1),
                                            transforms.ToTensor(), transforms.Normalize(norm_mean, norm_std)])
    val_transform = transforms.Compose([transforms.Resize((input_size,input_size)), transforms.ToTensor(),
                                        transforms.Normalize(norm_mean, norm_std)])

    training_set = HAM10000(df_train, transform=train_transform)
    train_loader = DataLoader(training_set, batch_size=32, shuffle=True, num_workers=1)
    validation_set = HAM10000(df_val, transform=train_transform)
    val_loader = DataLoader(validation_set, batch_size=32, shuffle=False, num_workers=1)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss().to(device)

    total_loss_train, total_acc_train = [],[]
    def train(train_loader, model, criterion, optimizer, epoch):
        model.train()
        train_loss = AverageMeter()
        train_acc = AverageMeter()
        curr_iter = (epoch - 1) * len(train_loader)
        for i, data in enumerate(train_loader):
            images, labels = data
            N = images.size(0)
            # print('image shape:',images.size(0), 'label shape',labels.size(0))
            images = Variable(images).to(device)
            labels = Variable(labels).to(device)

            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            prediction = outputs.max(1, keepdim=True)[1]
            train_acc.update(prediction.eq(labels.view_as(prediction)).sum().item()/N)
            train_loss.update(loss.item())
            curr_iter += 1
            if (i + 1) % 100 == 0:
                print('[epoch %d], [iter %d / %d], [train loss %.5f], [train acc %.5f]' % (
                    epoch, i + 1, len(train_loader), train_loss.avg, train_acc.avg))
                total_loss_train.append(train_loss.avg)
                total_acc_train.append(train_acc.avg)
        return train_loss.avg, train_acc.avg
    def validate(val_loader, model, criterion, optimizer, epoch):
        model.eval()
        val_loss = AverageMeter()
        val_acc = AverageMeter()
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                images, labels = data
                N = images.size(0)
                images = Variable(images).to(device)
                labels = Variable(labels).to(device)

                outputs = model(images)
                prediction = outputs.max(1, keepdim=True)[1]

                val_acc.update(prediction.eq(labels.view_as(prediction)).sum().item()/N)

                val_loss.update(criterion(outputs, labels).item())

        print('------------------------------------------------------------')
        print('[epoch %d], [val loss %.5f], [val acc %.5f]' % (epoch, val_loss.avg, val_acc.avg))
        print('------------------------------------------------------------')
        return val_loss.avg, val_acc.avg

    epoch_num = 2
    best_val_acc = 0
    total_loss_val, total_acc_val = [],[]
    for epoch in range(1, epoch_num+1):
        loss_train, acc_train = train(train_loader, model, criterion, optimizer, epoch)
        loss_val, acc_val = validate(val_loader, model, criterion, optimizer, epoch)
        total_loss_val.append(loss_val)
        total_acc_val.append(acc_val)
        if acc_val > best_val_acc:
            best_val_acc = acc_val
            print('*****************************************************')
            print('best record: [epoch %d], [val loss %.5f], [val acc %.5f]' % (epoch, loss_val, acc_val))
            print('*****************************************************')
    fig = plt.figure(num = 2)
    fig1 = fig.add_subplot(2,1,1)
    fig2 = fig.add_subplot(2,1,2)
    fig1.plot(total_loss_train, label = 'training loss')
    fig1.plot(total_acc_train, label = 'training accuracy')
    fig2.plot(total_loss_val, label = 'validation loss')
    fig2.plot(total_acc_val, label = 'validation accuracy')
    plt.legend()
    plt.show()

    def plot_confusion_matrix(cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

    model.eval()
    y_label = []
    y_predict = []
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            images, labels = data
            N = images.size(0)
            images = Variable(images).to(device)
            outputs = model(images)
            prediction = outputs.max(1, keepdim=True)[1]
            y_label.extend(labels.cpu().numpy())
            y_predict.extend(np.squeeze(prediction.cpu().numpy().T))

    # compute the confusion matrix
    confusion_mtx = confusion_matrix(y_label, y_predict)
    # plot the confusion matrix
    plot_labels = ['akiec', 'bcc', 'bkl', 'df', 'nv', 'vasc','mel']
    plot_confusion_matrix(confusion_mtx, plot_labels)

    # Generate a classification report
    report = classification_report(y_label, y_predict, target_names=plot_labels)
    print(report)

    label_frac_error = 1 - np.diag(confusion_mtx) / np.sum(confusion_mtx, axis=1)
    plt.bar(np.arange(7),label_frac_error)
    plt.xlabel('True Label')
    plt.ylabel('Fraction classified incorrectly')
if __name__ == '__main__':
    main()
