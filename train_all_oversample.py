import os
import pandas as pd
from PIL import Image
import time
from collections import defaultdict, Counter
import json
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.model_selection import train_test_split

class ImageDatasetCategories(Dataset):
    def __init__(self, csv_file, img_dir, image_column, category_column, transform=None):
        data = pd.read_csv(csv_file)
        self.data = data[data[category_column].notna() & (data[category_column] != "")]
        self.img_dir = img_dir
        self.transform = transform
        self.image_column = image_column
        self.category_column = category_column

        self.classes = sorted(self.data[self.category_column].unique())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.img_dir, row[self.image_column])
        image = Image.open(img_path).convert("RGB")

        label = self.class_to_idx[row[self.category_column]]

        if self.transform:
            image = self.transform(image)

        return image, label

def stratified_split(dataset, train_ratio=0.8):
    targets = [label for _, label in dataset]

    train_indices, test_indices = train_test_split(
        range(len(targets)),
        train_size=train_ratio,
        stratify=targets,
        random_state=42
    )

    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)

    return train_dataset, test_dataset

def train_model(train_loader, model, criterion, optimizer, device, train_size, num_epochs=10):
    epoch_losses = []
    epoch_accuracies = []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / train_size
        epoch_acc = 100.0 * correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%")
        epoch_losses.append(epoch_loss)
        epoch_accuracies.append(epoch_acc)
    print("Training finished")
    return model, epoch_losses, epoch_accuracies


def test_model(model, test_loader, device, criterion, classes):
    model.eval()
    correct, total = 0, 0
    test_loss = 0.0

    class_correct = [0] * len(classes)
    class_total = [0] * len(classes)
    
    misclassifications = defaultdict(int)
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            for i in range(len(labels)):
                label = labels[i].item()
                pred = predicted[i].item()
                class_total[label] += 1
                if pred == label:
                    class_correct[label] += 1
                else:
                    misclassifications[f'{classes[label]}_{classes[pred]}'] += 1
                    
    test_loss = test_loss / total
    test_acc = 100.0 * correct / total
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

    print("Per-class accuracy:")
    for i, cls in enumerate(classes):
        if class_total[i] > 0:
            acc = 100.0 * class_correct[i] / class_total[i]
            print(f"{str(cls):17s}: {acc:.2f}% ({class_correct[i]}/{class_total[i]})")
        else:
            print(f"{str(cls):17s}: No samples")

    if misclassifications:
        print("Misclassification summary:")
        for missclass, count in misclassifications.items():
            true_cls, pred_cls = missclass.split('_')
            print(f"{true_cls} classified as {pred_cls}: {count}")
    return test_loss, test_acc, misclassifications, class_correct, class_total

csv_file = "Yugi_db_with_categories_v2.csv"
img_dir = "Yugi_images_processed"
split_ratio = 0.8
batch_size = 32
lr = 0.001
workers = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

classification_category = ['CardCategory', 'Level', 'Property', 'Attribute']
epochs_train = 10

transform = transforms.Compose([
    transforms.Resize((180, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

model_names = ['resnet18', 'vgg16', 'densenet121', 'efficientnet_b0', 'mobilenet_v2']

training_losses = {}
training_accuracies = {}
test_accuracies = {}
test_losses = {}
all_missclassifications = {}
all_classes_correct = {}
all_classes_total = {}
all_classes = {}
training_times = {}
start_time_all = time.time()
for category in classification_category:
    print(f"\n\nTraining for category: {category}")
    print("Preparing dataset...")
    dataset = ImageDatasetCategories(csv_file, img_dir, 'Image_name', category, transform=transform)
    train_dataset, test_dataset = stratified_split(dataset)
    # start oversampling difference
    targets = [dataset[i][1] for i in train_dataset.indices]
    class_counts = np.bincount(targets)
    class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
    sample_weights = [class_weights[label] for label in targets]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    print("Counting samples per class in one epoch of oversampled train_loader...")
    label_counts = Counter()
    for images, labels in train_loader:
        for label in labels:
            label_counts[label.item()] += 1

    for idx, count in sorted(label_counts.items()):
        print(f"Class {idx} ({dataset.classes[idx]}): {count} samples in one epoch")
    label_counts_named = {dataset.classes[idx]: count for idx, count in label_counts.items()}

    with open(f"oversample_results/label_counts_{category}_oversample.json", "w") as file:
        json.dump(label_counts_named, file)
    # end oversampling difference
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print("Creating models...")
    num_classes = len(dataset.classes)

    all_models = []
    all_optimizers = []
    all_criterions = []
    
    all_models.append(models.resnet18(weights=None))
    all_models[0].fc = nn.Linear(all_models[0].fc.in_features, num_classes)
    all_models[0] = all_models[0].to(device)
    all_criterions.append(nn.CrossEntropyLoss())
    all_optimizers.append(optim.Adam(all_models[0].parameters(), lr=lr))
    
    all_models.append(models.vgg16(weights=None))
    all_models[1].classifier[6] = nn.Linear(all_models[1].classifier[6].in_features, num_classes)
    all_models[1] = all_models[1].to(device)
    all_criterions.append(nn.CrossEntropyLoss())
    all_optimizers.append(optim.Adam(all_models[1].parameters(), lr=lr))
    
    all_models.append(models.densenet121(weights=None))
    all_models[2].classifier = nn.Linear(all_models[2].classifier.in_features, num_classes)
    all_models[2] = all_models[2].to(device)
    all_criterions.append(nn.CrossEntropyLoss())
    all_optimizers.append(optim.Adam(all_models[2].parameters(), lr=lr))
    
    all_models.append(models.efficientnet_b0(weights=None))
    all_models[3].classifier[1] = nn.Linear(all_models[3].classifier[1].in_features, num_classes)
    all_models[3] = all_models[3].to(device)
    all_criterions.append(nn.CrossEntropyLoss())
    all_optimizers.append(optim.Adam(all_models[3].parameters(), lr=lr))

    all_models.append(models.mobilenet_v2(weights=None))
    all_models[4].classifier[1] = nn.Linear(all_models[4].classifier[1].in_features, num_classes)
    all_models[4] = all_models[4].to(device)
    all_criterions.append(nn.CrossEntropyLoss())
    all_optimizers.append(optim.Adam(all_models[4].parameters(), lr=lr))

    for i in range(5):
        print(f"\nTraining model: {model_names[i]} for category: {category}")
        total_start = time.time()
        all_models[i], training_losses[f'{model_names[i]}_{category}'], training_accuracies[f'{model_names[i]}_{category}'] = train_model(train_loader, all_models[i], all_criterions[i], all_optimizers[i], device, len(train_dataset), num_epochs=epochs_train)
        total_end = time.time()
        training_times[f'{model_names[i]}_{category}'] = total_end - total_start
        print(f"Training time for {model_names[i]} on category {category}: {training_times[f'{model_names[i]}_{category}']:.2f} seconds, saving model...")
        torch.save(all_models[i].state_dict(), f'oversample_models/{model_names[i]}_{category}_oversample.pth')
        print("Testing model...")
        classes = dataset.classes
        all_classes[f'{model_names[i]}_{category}'] = classes
        test_losses[f'{model_names[i]}_{category}'], test_accuracies[f'{model_names[i]}_{category}'], all_missclassifications[f'{model_names[i]}_{category}'], all_classes_correct[f'{model_names[i]}_{category}'], all_classes_total[f'{model_names[i]}_{category}'] = test_model(all_models[i], test_loader, device, all_criterions[i], classes)

end_time_all = time.time()
print(f"\nTotal training and testing time for all models and categories: {end_time_all - start_time_all:.2f} seconds")

print("\nSaving results to JSON files...")
with open("oversample_results/training_losses_oversample.json", "w") as file:
        json.dump(training_losses, file)
with open("oversample_results/training_accuracies_oversample.json", "w") as file:
        json.dump(training_accuracies, file)
with open("oversample_results/test_accuracies_oversample.json", "w") as file:
        json.dump(test_accuracies, file)
with open("oversample_results/test_losses_oversample.json", "w") as file:
        json.dump(test_losses, file)
with open("oversample_results/all_missclassifications_oversample.json", "w") as file:
        json.dump(all_missclassifications, file)
with open("oversample_results/all_classes_correct_oversample.json", "w") as file:
        json.dump(all_classes_correct, file)
with open("oversample_results/all_classes_total_oversample.json", "w") as file:
        json.dump(all_classes_total, file)
with open("oversample_results/all_classes_oversample.json", "w") as file:
        json.dump(all_classes, file)
with open("oversample_results/training_times_oversample.json", "w") as file:
        json.dump(training_times, file)