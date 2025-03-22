import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import train
import hyperparameters as HP
from dataprocessor.datacomposer import getData
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from FMoW import get_fmow_train_test_loader
import clip
import os
from torch.utils.data import DataLoader, TensorDataset
from dataprocessor import flickr_data_process
from dataprocessor.Adience import get_adience_train_test_loader
import torchvision.transforms as transforms

def get_text_tokens(model, device, text_descriptions):
    # Process text descriptions
    text_tokens = clip.tokenize(text_descriptions)
    text_tokens = text_tokens.to(device)
    return text_tokens


def process_test_loader(testloader, grouper, device):
    """
    Process test_loader, convert metadata to labels and return new test_loader.
    """
    processed_data = []

    for b_x, _, metadata in tqdm(testloader):
        # Use grouper to convert metadata to labels
        b_y = grouper.metadata_to_group(metadata).to(device)
        processed_data.append((b_x, b_y, metadata))

    return DataLoader(processed_data, batch_size=testloader.batch_size, shuffle=False)


def evaluate_zeroshot(model, testloader, device, text_descriptions, preprocess):
    model.eval()
    correct = 0
    total = 0
    if HP.clip_model == "ViT-B/32":
        transform = Compose([
            Resize(224),
            CenterCrop(224),
            Normalize((0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711))
        ])
    elif HP.clip_model == "RN50":
        transform = Compose([
            Resize(256),
            CenterCrop(224),
            Normalize((0.48145466, 0.4578275, 0.40821073),
                      (0.26862954, 0.26130258, 0.27577711))
        ])
    text_tokens = get_text_tokens(model, device, text_descriptions)

    with torch.no_grad():
        for b_x, b_y in tqdm(testloader, desc="zero-shot evaluation"):
            processed_images = transform(b_x).to(device)
            image_features = model.encode_image(processed_images)
            text_features = model.encode_text(text_tokens)

            # Normalize features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # Calculate similarity
            similarity = (100.0 * image_features @ text_features.T)
            predictions = similarity.argmax(dim=-1)

            correct += (predictions == b_y.to(device)).sum().cpu().item()
            total += len(b_y)

    accuracy = 100 * correct / total
    print(f"Zero-shot CLIP accuracy on test set: {accuracy:.2f}%")
    return accuracy


class LogisticRegressionTorch(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LogisticRegressionTorch, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)
    

def extract_features(model, data_loader, device, transform=None):
    model.eval() 
    features = []
    with torch.no_grad(): 
        for batch_idx, (images, _) in tqdm(enumerate(data_loader), desc='Extracting features'):
            if transform:
                images = [transform(img) for img in images] 
            images = torch.stack(tuple(images)).to(device) 

            # Extract image features
            batch_features = model.encode_image(images)
            features.append(batch_features.cpu())

            torch.cuda.empty_cache()  # Clear cache
    return torch.cat(features, dim=0).numpy()  # Merge all batch features and return


def train_classifier_torch(train_features, train_labels, test_features, test_labels, device):
    X_train = torch.FloatTensor(train_features).to(device)
    y_train = torch.LongTensor(train_labels).to(device)
    X_test = torch.FloatTensor(test_features).to(device)
    y_test = torch.LongTensor(test_labels).to(device)

    train_size = int(0.8 * len(X_train))
    val_size = len(X_train) - train_size

    indices = torch.randperm(len(X_train))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    X_train_split = X_train[train_indices]
    y_train_split = y_train[train_indices]
    X_val = X_train[val_indices]
    y_val = y_train[val_indices]

    lambda_range = np.logspace(-6, 6, 96)
    best_score = -1
    best_lambda = None
    best_model_state = None

    input_dim = train_features.shape[1]
    num_classes = len(torch.unique(y_train))

    for lambda_val in tqdm(lambda_range, desc="Searching best lambda"):
        model_classifier = LogisticRegressionTorch(
            input_dim, num_classes).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.LBFGS(
            model_classifier.parameters(), lr=1, max_iter=100)

        X_train_split = X_train_split.to(device)
        y_train_split = y_train_split.to(device)
        X_val = X_val.to(device)
        y_val = y_val.to(device)

        def closure():
            optimizer.zero_grad()
            outputs = model_classifier(X_train_split)
            loss = criterion(outputs, y_train_split)
            l2_reg = torch.tensor(0.).to(device)
            for param in model_classifier.parameters():
                l2_reg += torch.norm(param)**2
            loss += lambda_val * l2_reg
            loss.backward()
            return loss

        optimizer.step(closure)

        model_classifier.eval()
        with torch.no_grad():
            val_outputs = model_classifier(X_val)
            _, predicted = torch.max(val_outputs, 1)
            val_score = (predicted == y_val).float().mean().item()

        if val_score > best_score:
            best_score = val_score
            best_lambda = lambda_val
            best_model_state = model_classifier.state_dict()

    final_model = LogisticRegressionTorch(input_dim, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.LBFGS(final_model.parameters(), lr=1, max_iter=100)

    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)

    def final_closure():
        optimizer.zero_grad()
        outputs = final_model(X_train)
        loss = criterion(outputs, y_train)
        l2_reg = torch.tensor(0.).to(device)
        for param in final_model.parameters():
            l2_reg += torch.norm(param)**2
        loss += best_lambda * l2_reg
        loss.backward()
        return loss

    optimizer.step(final_closure)

    final_model.eval()
    with torch.no_grad():
        test_outputs = final_model(X_test)
        _, predicted = torch.max(test_outputs, 1)
        test_accuracy = (predicted == y_test).float().mean().item()
    return test_accuracy, best_lambda


def train_lp(model, train_loader, test_loader, device, batch_size, transform=None):
    train_features = extract_features(model, train_loader, device, transform)
    test_features = extract_features(model, test_loader, device, transform)

    train_labels = [label for _, label in train_loader]
    test_labels = [label for _, label in test_loader]

    train_labels = torch.cat(train_labels, dim=0).numpy()
    test_labels = torch.cat(test_labels, dim=0).numpy()

    test_accuracy, best_lambda = train_classifier_torch(
        train_features, train_labels, test_features, test_labels, device)
    return test_accuracy, best_lambda


def get_cifar_loaders(batch_size=128):
    def _remap_labels(labels):
        return [0 if x in {0, 1, 8, 9} else 1 for x in labels]

    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2470, 0.2435, 0.2616]

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),  
        transforms.RandomHorizontalFlip(p=0.5),               
        transforms.ToTensor(),                                
        transforms.Normalize(mean, std)                      
    ])

    train_set = CIFAR10(root='data/', train=True,
                        download=True, transform=None)

    train_set.targets = _remap_labels(train_set.targets)

    Xs, Ys = [], []
    for img, label in train_set:  
        aug1 = train_transform(img)  
        aug2 = train_transform(img)  

        Xs.extend([aug1, aug2])
        Ys.extend([label, label])

    train_tensors = TensorDataset(
        torch.stack(Xs), 
        torch.LongTensor(Ys)
    )

    train_loader = DataLoader(
        train_tensors,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True
    )

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    test_set = CIFAR10(root='data/', train=False,
                       download=True, transform=test_transform)
    test_set.targets = _remap_labels(test_set.targets)

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    return train_loader, test_loader


def get_transform():
    """Get the appropriate transform based on the CLIP model."""
    if HP.clip_model == "ViT-B/32":
        transform = Compose([
            Resize(224),
            CenterCrop(224),
            Normalize((0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711))
        ])
    elif HP.clip_model == "RN50":
        transform = Compose([
            Resize(256),
            CenterCrop(224),
            Normalize((0.48145466, 0.4578275, 0.40821073),
                      (0.26862954, 0.26130258, 0.27577711))
        ])
    else:
        raise ValueError("Invalid CLIP model name")
    
    return transform


def get_text_descriptions(dataset_name):
    """Get text descriptions for different datasets."""
    if dataset_name == 'FMoW':
        return [
            "an aerial or satellite image of a location in Asia, showing Asian architectural styles and landscapes",
            "an aerial or satellite image of a location in Europe, featuring European urban planning and architecture",
            "an aerial or satellite image of a location in Africa, displaying African terrain and settlements",
            "an aerial or satellite image of a location in the Americas, showing North or South American buildings and landscapes",
            "an aerial or satellite image of a location in Oceania, featuring Pacific island or Australian landscapes",
            "an aerial or satellite image of other global regions or unclear geographical features"
        ]
    elif dataset_name == 'beauty':
        return [
            "an image of extremely low quality, out of focus, underexposed, and badly framed",
            "an image of low quality with technical flaws such as slight blur, slight over/underexposure, and incorrect framing, with no artistic value",
            "an image of standard quality with no technical flaws, subject well framed, in focus, easily recognizable, but without artistic value",
            "a professional-quality image with flawless framing, focus, and lighting, or with some artistic value",
            "a very appealing image showing both outstanding professional quality and high artistic value, including photographic and/or editing techniques"
        ]
    elif dataset_name == 'adience':
        return [
            "A young adult aged between 21 and 35 years",
            "An infant aged between 0 and 6 years",
            "An adult aged between 36 and 55 years",
            "A child aged between 7 and 12 years",
            "A teenager aged between 13 and 20 years",
            "A senior aged 56 years or above"
        ]
    else:
        raise ValueError(f"No text descriptions available for dataset: {dataset_name}")


if __name__ == "__main__":
    # Note: You might want to review this line as it could contain personal information
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Set CUDA visible devices
    print("CUDA_VISIBLE_DEVICES:", os.environ["CUDA_VISIBLE_DEVICES"])
    
    # Choose device: GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the CLIP model
    model, preprocess = clip.load(HP.clip_model, device=device)
    model = model.to(device)
    print("Model:", HP.clip_model)
    
    # Get transform for the specific model
    transform = get_transform()
    
    # Load the dataset
    print('Dataset:', HP.data_set)
    
    # Load appropriate dataset
    if HP.data_set == 'FMoW':
        train_loader, test_loader = get_fmow_train_test_loader()
    elif HP.data_set == 'beauty':
        train_loader, test_loader = flickr_data_process.get_train_test_loader(HP.batch_size)
    elif HP.data_set == 'adience':
        train_loader, test_loader = get_adience_train_test_loader(batchsize=HP.batch_size)
    elif HP.data_set == 'cifar10-2':
        train_loader, test_loader = get_cifar_loaders(HP.batch_size)
    elif HP.data_set == 'beauty-ori':
        train_loader, test_loader = flickr_data_process.get_original_label_train_test_loader(HP.batch_size)
    else:
        raise ValueError("Invalid dataset name")
    
    # Choose the mode: zero-shot or linear probing
    if HP.clip_mode == 'zero-shot':
        # Get text descriptions for the dataset
        text_descriptions = get_text_descriptions(HP.data_set)
        
        # Evaluate with zero-shot
        accuracy = evaluate_zeroshot(model, test_loader, device, text_descriptions, preprocess)
        print(f"Zero-shot accuracy: {accuracy:.2f}%")
    
    elif HP.clip_mode == 'linear-probe':
        # Train linear probe classifier
        test_accuracy, best_lambda = train_lp(model, train_loader, test_loader, device, HP.batch_size, transform)
        print(f"Linear probe accuracy: {test_accuracy:.2f}%")
        print(f"Best lambda: {best_lambda:.2e}")
    
    else:
        raise ValueError("Invalid clip_mode. Use 'zero-shot' or 'linear-probe'")