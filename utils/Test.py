# Complete pipeline for fine-tuning EfficientNet on ORBIT Dataset
import os
import json
import random
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.transforms import functional as TF

# For reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed()

# Configuration
class Config:
    data_root = "path/to/orbit_dataset"  # Change this to your dataset path
    metadata_file = "metadata.json"
    frame_dir = "frames"
    
    # Few-shot learning parameters
    n_way = 5  # Number of classes per episode
    k_shot = 5  # Number of support examples per class
    n_query = 15  # Number of query examples per class
    episodes_per_epoch = 100
    
    # Training parameters
    batch_size = 16
    learning_rate = 0.0001
    num_epochs = 30
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Model parameters
    pretrained = True
    efficientnet_version = "b0"  # Options: b0-b7
    
    # Paths
    save_dir = "orbit_efficientnet_models"
    log_dir = "orbit_efficientnet_logs"
    
config = Config()

# Create directories
os.makedirs(config.save_dir, exist_ok=True)
os.makedirs(config.log_dir, exist_ok=True)

# Data preparation utilities
class ORBITDataset:
    def __init__(self, data_root, split="train"):
        """
        Initialize ORBIT dataset handler
        
        Args:
            data_root: Path to the ORBIT dataset
            split: One of 'train', 'val', or 'test'
        """
        self.data_root = data_root
        self.split = split
        self.metadata = self._load_metadata()
        self.users = self._get_users_for_split()
        self.object_classes = self._get_object_classes()
        
        # Map users to their objects and frames
        self.user_objects = self._map_users_to_objects()
        self.frame_paths = self._generate_frame_paths()
        
        # Transform for preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])
        ])
        
    def _load_metadata(self):
        """Load and parse the metadata JSON file"""
        metadata_path = os.path.join(self.data_root, Config.metadata_file)
        with open(metadata_path, 'r') as f:
            return json.load(f)
    
    def _get_users_for_split(self):
        """Extract list of users for the current split"""
        return [user for user, data in self.metadata.items() 
                if data.get('split') == self.split]
    
    def _get_object_classes(self):
        """Extract all unique object classes from the dataset"""
        objects = set()
        for user, data in self.metadata.items():
            if data.get('split') == self.split:
                for obj in data.get('objects', {}).keys():
                    objects.add(obj)
        return list(objects)
    
    def _map_users_to_objects(self):
        """Create mapping from users to their objects"""
        user_objects = {}
        for user in self.users:
            user_objects[user] = list(self.metadata[user]['objects'].keys())
        return user_objects
    
    def _generate_frame_paths(self):
        """Generate paths to all frame images"""
        frame_paths = defaultdict(lambda: defaultdict(list))
        
        for user in self.users:
            user_dir = os.path.join(self.data_root, user)
            for obj in self.user_objects[user]:
                obj_dir = os.path.join(user_dir, obj, Config.frame_dir)
                
                # Get protocol directories (clean, clutter, etc.)
                if not os.path.exists(obj_dir):
                    continue
                    
                protocol_dirs = [d for d in os.listdir(obj_dir) 
                                if os.path.isdir(os.path.join(obj_dir, d))]
                
                for protocol in protocol_dirs:
                    protocol_dir = os.path.join(obj_dir, protocol)
                    
                    # Get video directories
                    video_dirs = [v for v in os.listdir(protocol_dir) 
                                 if os.path.isdir(os.path.join(protocol_dir, v))]
                    
                    for video in video_dirs:
                        video_dir = os.path.join(protocol_dir, video)
                        # Add frame paths - keeping only training videos
                        if self.metadata[user]['objects'][obj].get('train', False):
                            frames = [os.path.join(video_dir, f) for f in os.listdir(video_dir) 
                                     if f.endswith(('.jpg', '.png'))]
                            frame_paths[user][obj].extend(frames)
        
        return frame_paths
    
    def sample_episode(self, n_way=None, k_shot=None, n_query=None):
        """
        Sample a few-shot learning episode from the dataset
        
        Args:
            n_way: Number of classes per episode
            k_shot: Number of support examples per class
            n_query: Number of query examples per class
            
        Returns:
            Dictionary containing support and query sets
        """
        if n_way is None:
            n_way = Config.n_way
        if k_shot is None:
            k_shot = Config.k_shot
        if n_query is None:
            n_query = Config.n_query
            
        # Sample users
        sampled_users = random.sample(self.users, min(n_way, len(self.users)))
        
        support_images = []
        support_labels = []
        query_images = []
        query_labels = []
        
        for label_idx, user in enumerate(sampled_users):
            # For each user, sample one object
            available_objects = [obj for obj in self.user_objects[user] 
                               if len(self.frame_paths[user][obj]) >= (k_shot + n_query)]
            
            if not available_objects:
                continue
                
            obj = random.choice(available_objects)
            
            # Sample frames for this object
            available_frames = self.frame_paths[user][obj]
            sampled_frames = random.sample(available_frames, k_shot + n_query)
            
            # Split into support and query
            support_frames = sampled_frames[:k_shot]
            query_frames = sampled_frames[k_shot:k_shot + n_query]
            
            # Load images
            for frame_path in support_frames:
                img = self._load_and_transform_image(frame_path)
                support_images.append(img)
                support_labels.append(label_idx)
                
            for frame_path in query_frames:
                img = self._load_and_transform_image(frame_path)
                query_images.append(img)
                query_labels.append(label_idx)
        
        # Convert to tensors
        support_images = torch.stack(support_images)
        support_labels = torch.tensor(support_labels)
        query_images = torch.stack(query_images)
        query_labels = torch.tensor(query_labels)
        
        return {
            'support_images': support_images,
            'support_labels': support_labels,
            'query_images': query_images,
            'query_labels': query_labels
        }
    
    def _load_and_transform_image(self, image_path):
        """Load and preprocess an image"""
        try:
            img = Image.open(image_path).convert('RGB')
            return self.transform(img)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a blank image as fallback
            blank = Image.new('RGB', (224, 224), color=(0, 0, 0))
            return self.transform(blank)

# EfficientNet-based Prototypical Network for Few-Shot Learning
class EfficientNetProtoNet(nn.Module):
    def __init__(self, version='b0', pretrained=True, num_classes=1000):
        super(EfficientNetProtoNet, self).__init__()
        
        # Load the appropriate EfficientNet model
        if version == 'b0':
            self.efficientnet = models.efficientnet_b0(pretrained=pretrained)
        elif version == 'b1':
            self.efficientnet = models.efficientnet_b1(pretrained=pretrained)
        elif version == 'b2':
            self.efficientnet = models.efficientnet_b2(pretrained=pretrained)
        elif version == 'b3':
            self.efficientnet = models.efficientnet_b3(pretrained=pretrained)
        elif version == 'b4':
            self.efficientnet = models.efficientnet_b4(pretrained=pretrained)
        elif version == 'b5':
            self.efficientnet = models.efficientnet_b5(pretrained=pretrained)
        elif version == 'b6':
            self.efficientnet = models.efficientnet_b6(pretrained=pretrained)
        elif version == 'b7':
            self.efficientnet = models.efficientnet_b7(pretrained=pretrained)
        else:
            raise ValueError(f"Unknown EfficientNet version: {version}")
        
        # Remove classifier for feature extraction
        feature_dims = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier = nn.Identity()
        
        # Embedding projection layer (reduce dimensionality)
        self.embedding = nn.Linear(feature_dims, 512)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.embedding.weight)
        
    def forward(self, x):
        """Extract features from input images"""
        features = self.efficientnet(x)
        embeddings = self.embedding(features)
        # L2 normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings
    
    def predict(self, support_images, support_labels, query_images):
        """
        Perform episodic prediction using prototypical networks approach
        
        Args:
            support_images: Images in the support set
            support_labels: Labels for the support set
            query_images: Query images to classify
            
        Returns:
            Predicted class probabilities for query images
        """
        # Extract features
        support_features = self(support_images)
        query_features = self(query_images)
        
        # Calculate prototypes (class centroids)
        unique_labels = torch.unique(support_labels)
        prototypes = torch.zeros(len(unique_labels), support_features.shape[1]).to(support_features.device)
        
        for i, label in enumerate(unique_labels):
            mask = support_labels == label
            prototypes[i] = support_features[mask].mean(0)
        
        # Calculate distances to prototypes
        dists = torch.cdist(query_features, prototypes)
        
        # Convert distances to probabilities (negative distance)
        logits = -dists
        probs = F.softmax(logits, dim=1)
        
        return probs

# Training and evaluation functions
def train_episode(model, optimizer, episode_data, device):
    """Train model on a single episode"""
    model.train()
    
    # Move data to device
    support_images = episode_data['support_images'].to(device)
    support_labels = episode_data['support_labels'].to(device)
    query_images = episode_data['query_images'].to(device)
    query_labels = episode_data['query_labels'].to(device)
    
    # Forward pass
    optimizer.zero_grad()
    
    # Get features
    support_features = model(support_images)
    query_features = model(query_images)
    
    # Calculate prototypes for each class
    unique_labels = torch.unique(support_labels)
    n_classes = len(unique_labels)
    prototypes = torch.zeros(n_classes, support_features.shape[1]).to(device)
    
    for i, label in enumerate(unique_labels):
        mask = support_labels == label
        prototypes[i] = support_features[mask].mean(0)
    
    # Calculate distance from queries to prototypes
    dists = torch.cdist(query_features, prototypes)
    
    # Convert to logits and calculate loss
    logits = -dists
    loss = F.cross_entropy(logits, query_labels)
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    # Calculate accuracy
    _, preds = torch.max(logits, dim=1)
    acc = (preds == query_labels).float().mean().item()
    
    return loss.item(), acc

def evaluate(model, dataset, num_episodes=100, device=None):
    """Evaluate model on multiple episodes"""
    if device is None:
        device = Config.device
        
    model.eval()
    
    total_loss = 0
    total_acc = 0
    
    with torch.no_grad():
        for i in range(num_episodes):
            # Sample episode
            episode_data = dataset.sample_episode()
            
            # Move data to device
            support_images = episode_data['support_images'].to(device)
            support_labels = episode_data['support_labels'].to(device)
            query_images = episode_data['query_images'].to(device)
            query_labels = episode_data['query_labels'].to(device)
            
            # Get features
            support_features = model(support_images)
            query_features = model(query_images)
            
            # Calculate prototypes
            unique_labels = torch.unique(support_labels)
            n_classes = len(unique_labels)
            prototypes = torch.zeros(n_classes, support_features.shape[1]).to(device)
            
            for j, label in enumerate(unique_labels):
                mask = support_labels == label
                prototypes[j] = support_features[mask].mean(0)
            
            # Calculate distance and loss
            dists = torch.cdist(query_features, prototypes)
            logits = -dists
            loss = F.cross_entropy(logits, query_labels)
            
            # Calculate accuracy
            _, preds = torch.max(logits, dim=1)
            acc = (preds == query_labels).float().mean().item()
            
            total_loss += loss.item()
            total_acc += acc
    
    avg_loss = total_loss / num_episodes
    avg_acc = total_acc / num_episodes
    
    return avg_loss, avg_acc

# Main training loop
def train_model():
    print("Preparing datasets...")
    train_dataset = ORBITDataset(Config.data_root, split="train")
    val_dataset = ORBITDataset(Config.data_root, split="val")
    
    print("Initializing model...")
    model = EfficientNetProtoNet(
        version=Config.efficientnet_version, 
        pretrained=Config.pretrained
    )
    model = model.to(Config.device)
    
    optimizer = optim.Adam(model.parameters(), lr=Config.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    
    best_val_acc = 0
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    print(f"Starting training on {Config.device}...")
    for epoch in range(Config.num_epochs):
        # Training phase
        epoch_loss = 0
        epoch_acc = 0
        
        for episode in tqdm(range(Config.episodes_per_epoch), desc=f"Epoch {epoch+1}/{Config.num_epochs}"):
            episode_data = train_dataset.sample_episode()
            loss, acc = train_episode(model, optimizer, episode_data, Config.device)
            epoch_loss += loss
            epoch_acc += acc
        
        avg_train_loss = epoch_loss / Config.episodes_per_epoch
        avg_train_acc = epoch_acc / Config.episodes_per_epoch
        train_losses.append(avg_train_loss)
        train_accs.append(avg_train_acc)
        
        # Validation phase
        val_loss, val_acc = evaluate(model, val_dataset)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, os.path.join(Config.save_dir, 'best_model.pth'))
            print(f"Saved new best model with validation accuracy: {val_acc:.4f}")
    
    # Save final model
    torch.save({
        'epoch': Config.num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
    }, os.path.join(Config.save_dir, 'final_model.pth'))
    
    # Plot training curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(Config.log_dir, 'training_curves.png'))
    
    return model

# Function to test model on test set
def test_model(model_path):
    print("Loading test dataset...")
    test_dataset = ORBITDataset(Config.data_root, split="test")
    
    print("Loading model...")
    model = EfficientNetProtoNet(
        version=Config.efficientnet_version, 
        pretrained=False
    )
    
    checkpoint = torch.load(model_path, map_location=Config.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(Config.device)
    
    print("Evaluating on test set...")
    test_loss, test_acc = evaluate(model, test_dataset, num_episodes=200)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
    
    return test_loss, test_acc

# Entry point
if __name__ == "__main__":
    print("Starting ORBIT dataset fine-tuning with EfficientNet...")
    model = train_model()
    
    # Test the best model
    print("\nTesting best model...")
    test_model(os.path.join(Config.save_dir, 'best_model.pth'))
