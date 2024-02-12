from torchvision import datasets
from torch.utils.data import Subset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

image_size = 224
batch_size = 32
num_clients = 3

# Preprocessing
transforms = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dataset = datasets.ImageFolder('Overall-Dataset', transform=transforms)

classes = dataset.classes

def train_test_ds(data, test_split=0.1):
    train_idx, val_idx = train_test_split(list(range(len(data))), test_size=test_split)
    train_data = Subset(data, train_idx)
    test_data = Subset(data, val_idx)

    return train_data, test_data


train_data, test_data = train_test_ds(dataset)

# DataLoaders
test_loader = DataLoader(test_data, batch_size=batch_size, pin_memory=True)

# Client Data Loaders
c_train_loaders = []
c_val_loaders = []
data_loader_size = len(train_data) // num_clients

for i in range(num_clients):
    start_idx = i * data_loader_size
    end_idx = (i + 1) * data_loader_size if i < num_clients - 1 else len(train_data)

    subset = Subset(train_data, list(range(start_idx, end_idx)))
    loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
    c_train_loaders.append(loader)

    val_loader_size = (len(subset)*2)//9
    start_idx = len(subset) - val_loader_size
    end_idx = len(subset)

    subset = Subset(subset, list(range(start_idx, end_idx)))
    loader = DataLoader(subset, batch_size, shuffle=True)
    c_val_loaders.append(loader)