import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.semi_supervised import LabelPropagation

class MetaLearningIDS:
    def __init__(self, dataset_paths, input_size, hidden_size=128, num_tasks=4):
        self.dataset_paths = dataset_paths
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_tasks = num_tasks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.scaler = StandardScaler()
        self.label_binarizer = MultiLabelBinarizer()
        self.meta_learner = MetaLearner(input_size, hidden_size, num_tasks).to(self.device)
    
    def _load_and_preprocess(self):
        dfs = [pd.read_csv(path) for path in self.dataset_paths]
        full_df = pd.concat(dfs, axis=0).dropna().reset_index(drop=True)
        
        full_df['labels'] = full_df['Label'].apply(lambda x: [x] if x != 'BENIGN' else [])
        labeled_mask = full_df['labels'].apply(lambda x: len(x) > 0)
        X = full_df.drop(columns=['labels', 'Label'], errors='ignore')
        y = full_df['labels']
        
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        y_bin = self.label_binarizer.fit_transform(y)
        
        return X_scaled, y_bin

    def _create_tasks(self, X, y):
        tasks = []
        for _ in range(self.num_tasks):
            X_train, _, y_train, _ = train_test_split(X, y, test_size=0.3)
            support, query, support_labels, query_labels = train_test_split(X_train, y_train, test_size=0.2)
            
            tasks.append({
                'support': (torch.FloatTensor(support).to(self.device), torch.FloatTensor(support_labels).to(self.device)),
                'query': (torch.FloatTensor(query).to(self.device), torch.FloatTensor(query_labels).to(self.device))
            })
        return tasks

    def train(self, meta_epochs=50, adaptation_steps=3):
        X, y = self._load_and_preprocess()
        tasks = self._create_tasks(X, y)
        
        trainer = MetaTrainingFramework(self.meta_learner, tasks, adaptation_steps)
        trainer.meta_train(meta_epochs)

    def evaluate(self, test_data):
        self.meta_learner.eval()
        X_test, y_test = test_data
        X_test = torch.FloatTensor(self.scaler.transform(X_test)).to(self.device)
        y_test = torch.FloatTensor(self.label_binarizer.transform(y_test)).to(self.device)
        
        with torch.no_grad():
            logits = self.meta_learner(X_test)
            preds = torch.sigmoid(logits) > 0.5
            acc = (preds == y_test).float().mean()
        
        return acc.item()

class MetaLearner(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, inner_lr=0.01, meta_lr=0.001):
        super(MetaLearner, self).__init__()
        self.base_model = MultiLabelMLP(input_size, hidden_size, num_classes)
        self.inner_lr = inner_lr
        self.meta_optimizer = optim.Adam(self.base_model.parameters(), lr=meta_lr)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, x):
        return self.base_model(x)

    def adapt(self, task_data, adaptation_steps):
        fast_weights = list(self.base_model.parameters())
        for _ in range(adaptation_steps):
            loss = self.compute_loss(task_data, fast_weights)
            grads = torch.autograd.grad(loss, fast_weights, create_graph=True)
            fast_weights = [w - self.inner_lr * g for w, g in zip(fast_weights, grads)]
        return fast_weights

    def compute_loss(self, data, weights=None):
        x, y = data
        logits = self.base_model(x) if weights is None else self._forward_with_weights(x, weights)
        return self.loss_fn(logits, y)

    def _forward_with_weights(self, x, weights):
        x = x.view(x.size(0), -1)
        for i, (name, param) in enumerate(self.base_model.named_parameters()):
            if 'weight' in name:
                x = F.linear(x, weights[i], weights[i+1])
            elif 'bias' in name:
                continue
            x = F.relu(x)
        return x

class MultiLabelMLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MultiLabelMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size//2, num_classes)
        )
    
    def forward(self, x):
        return self.net(x.view(x.size(0), -1))

class MetaTrainingFramework:
    def __init__(self, meta_learner, tasks, adaptation_steps):
        self.meta_learner = meta_learner
        self.tasks = tasks
        self.adaptation_steps = adaptation_steps

    def meta_train(self, num_epochs):
        for epoch in range(num_epochs):
            meta_loss = 0
            task_accuracies = []
            
            for task in self.tasks:
                fast_weights = self.meta_learner.adapt(task['support'], self.adaptation_steps)
                query_loss = self.meta_learner.compute_loss(task['query'], fast_weights)
                meta_loss += query_loss
                
                with torch.no_grad():
                    x_test, y_test = task['query']
                    logits = self.meta_learner._forward_with_weights(x_test, fast_weights)
                    preds = torch.sigmoid(logits) > 0.5
                    acc = (preds == y_test).float().mean()
                    task_accuracies.append(acc.item())
            
            self.meta_learner.meta_optimizer.zero_grad()
            meta_loss.backward()
            self.meta_learner.meta_optimizer.step()
            
            print(f"Epoch {epoch+1}/{num_epochs}, Meta Loss: {meta_loss.item()/len(self.tasks):.4f}, Avg Accuracy: {np.mean(task_accuracies):.4f}")

if __name__ == "__main__":
    ids_system = MetaLearningIDS(
        dataset_paths=["path/to/dataset1.csv", "path/to/dataset2.csv"],
        input_size=42,
        num_tasks=4
    )
    
    ids_system.train(meta_epochs=50, adaptation_steps=3)
    test_data = ...  # Load test dataset
    accuracy = ids_system.evaluate(test_data)
    print(f"Final Detection Accuracy: {accuracy*100:.2f}%")
