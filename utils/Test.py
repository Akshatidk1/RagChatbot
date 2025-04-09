import os
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import pandas as pd
from PIL import Image
import numpy as np
from tqdm import tqdm

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} for training")

class ORBITDataset(Dataset):
    def __init__(self, root_dir, users, mode='train', transform=None):
        """
        Args:
            root_dir (str): Path to the ORBIT dataset
            users (list): List of users to include
            mode (str): 'train', 'val', or 'test'
            transform: Optional transform to be applied to samples
        """
        self.root_dir = root_dir
        self.users = users
        self.mode = mode
        self.transform = transform
        
        self.samples = []
        self.labels = []
        self.label_map = {}  # Maps object names to label indices
        
        self._load_data()
    
    def _load_data(self):
        label_idx = 0
        
        # For each user
        for user in self.users:
            user_dir = os.path.join(self.root_dir, user)
            if not os.path.isdir(user_dir):
                continue
                
            # For each object in user's collection
            objects = [obj for obj in os.listdir(user_dir) if os.path.isdir(os.path.join(user_dir, obj))]
            for obj in objects:
                obj_dir = os.path.join(user_dir, obj)
                
                # Map this object to a label index
                obj_key = f"{user}_{obj}"
                if obj_key not in self.label_map:
                    self.label_map[obj_key] = label_idx
                    label_idx += 1
                
                # Get all clean/train and/or clutter/test videos as specified by mode
                if self.mode == 'train':
                    videos = [v for v in os.listdir(obj_dir) if "clean" in v.lower()]
                elif self.mode == 'val':
                    # Use some clean videos for validation
                    all_clean = [v for v in os.listdir(obj_dir) if "clean" in v.lower()]
                    videos = all_clean[:max(1, len(all_clean) // 5)]  # Use 20% for validation
                elif self.mode == 'test':
                    videos = [v for v in os.listdir(obj_dir) if "clutter" in v.lower()]
                
                # For each video, get frames
                for video in videos:
                    video_dir = os.path.join(obj_dir, video)
                    if not os.path.isdir(video_dir):
                        continue
                    
                    frames = [f for f in os.listdir(video_dir) if f.endswith(('.jpg', '.png'))]
                    # For training, use all frames. For validation/testing, sample frames
                    if self.mode == 'train':
                        selected_frames = frames
                    else:
                        # Sample a subset of frames for val/test to keep it manageable
                        selected_frames = frames[::max(1, len(frames) // 10)]
                    
                    for frame in selected_frames:
                        self.samples.append(os.path.join(video_dir, frame))
                        self.labels.append(self.label_map[obj_key])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.labels[idx]
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def load_user_splits(root_dir, split_file=None):
    """
    Load user splits from the ORBIT dataset.
    If split_file is not provided, creates a random split.
    """
    if split_file and os.path.exists(split_file):
        # Load predefined splits
        splits = pd.read_csv(split_file)
        train_users = splits[splits['split'] == 'train']['user'].tolist()
        val_users = splits[splits['split'] == 'val']['user'].tolist()
        test_users = splits[splits['split'] == 'test']['user'].tolist()
    else:
        # Create random splits
        users = [user for user in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, user))]
        np.random.shuffle(users)
        
        train_size = int(0.7 * len(users))
        val_size = int(0.15 * len(users))
        
        train_users = users[:train_size]
        val_users = users[train_size:train_size + val_size]
        test_users = users[train_size + val_size:]
    
    return train_users, val_users, test_users

def get_model(num_classes):
    """
    Get a pre-trained EfficientNet-B0 model and modify the final layer
    for fine-tuning on the ORBIT dataset.
    """
    model = models.efficientnet_b0(pretrained=True)
    
    # Freeze the early layers
    for param in list(model.parameters())[:-10]:
        param.requires_grad = False
    
    # Replace final classifier
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    
    return model

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    """
    Train the model.
    """
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in tqdm(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        
        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        
        # Validation phase
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
        
        epoch_loss = running_loss / len(val_loader.dataset)
        epoch_acc = running_corrects.double() / len(val_loader.dataset)
        
        print(f'Val Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        
        # Save the best model
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model.state_dict(), 'best_efficientnet_orbit.pth')
            print(f'Saved best model with accuracy: {best_acc:.4f}')
    
    return model

def main():
    # Path to the ORBIT dataset
    orbit_dir = 'path/to/ORBIT/dataset'  # Update this path
    
    # Data transformations
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load user splits
    train_users, val_users, test_users = load_user_splits(orbit_dir)
    
    # Create datasets
    train_dataset = ORBITDataset(orbit_dir, train_users, mode='train', transform=train_transform)
    val_dataset = ORBITDataset(orbit_dir, val_users, mode='val', transform=val_transform)
    
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(val_dataset)}")
    print(f"Number of classes: {len(train_dataset.label_map)}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Create the model
    num_classes = len(train_dataset.label_map)
    model = get_model(num_classes)
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    model = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10)
    
    print("Training complete!")

if __name__ == "__main__":
    main()
