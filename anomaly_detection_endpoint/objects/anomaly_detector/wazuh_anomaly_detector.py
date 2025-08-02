from transformers import BertTokenizer, BertModel
import torch.optim as optim
from tqdm import tqdm
from torch import nn
import torch
import math

class WazuhAnomalyDetector(nn.Module):
    def __init__(self, bert_model_name: str = "bert-base-uncased", dim_feedforward: int = 1024, device: str = "cpu"):
        super(WazuhAnomalyDetector, self).__init__()
        self.device = device

        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.bert = BertModel.from_pretrained(bert_model_name)
        for param in self.bert.parameters():
            param.requires_grad = False

        d_model = self.bert.config.hidden_size

        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = None
        self.best_loss = float("inf")

        self.fc = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(dim_feedforward, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(dim_feedforward, 1),
        )

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state

        x = last_hidden_state.mean(dim=1)
        x = self.fc(x)
        return x.squeeze(-1)

    def save_pretrained(self, path):
        torch.save(self.state_dict(), path)

    def load_pretrained(self, path):
        """Load a pretrained model from the specified path."""
        self.load_state_dict(
            torch.load(
                path,
                map_location=self.device,
            )
        )
        self.to(self.device)

    def _collate_fn(self, batch):
        texts, labels = zip(*batch)
        return texts, labels

    def train_model(self, batch_size, n_epochs, lr, train_dataset, test_dataset, save_path="model.pt"):
        self.train()

        self.optimizer = optim.AdamW(self.parameters(), lr=lr)
        # self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.5)
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, collate_fn=self._collate_fn
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self._collate_fn,
        )

        for epoch in range(n_epochs):
            print(f"Epoch {epoch + 1}/{n_epochs}")
            total_loss = self._train_epoch(train_loader)
            print(f"Loss: {total_loss:.4f}")

            val_loss = self._validate_epoch(test_loader)
            print(f"Validation Loss: {val_loss:.4f}")
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                print(f"Saving best model in {save_path}")
                self.save_pretrained(save_path)

    def _train_epoch(self, train_loader):
        self.train()
        total_loss = 0.0
    
        for batch in tqdm(train_loader):
            texts, labels = batch
            encodings = self.tokenizer(
                list(texts),
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            input_ids = encodings["input_ids"].to(self.device)
            attention_mask = encodings["attention_mask"].to(self.device)
            labels = torch.tensor(labels, dtype=torch.float, device=self.device)
    
            self.optimizer.zero_grad()
            outputs = self(input_ids, attention_mask)
            loss = self.criterion(outputs.view(-1), labels.float())
            loss.backward()
            self.optimizer.step()
            # self.scheduler.step()
    
            total_loss += loss.item()
    
        return total_loss / len(train_loader)


    def _validate_epoch(self, val_loader):
        self.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                texts, labels = batch
                encodings = self.tokenizer(
                    list(texts),
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                )
                input_ids = encodings["input_ids"].to(self.device)
                attention_mask = encodings["attention_mask"].to(self.device)
                labels = torch.tensor(labels, dtype=torch.float, device=self.device)

                outputs = self(input_ids, attention_mask)
                loss = self.criterion(outputs.squeeze(), labels.float())
                total_loss += loss.item()

        return total_loss / len(val_loader)

    @torch.no_grad()
    def predict(self, x):
        self.eval()
        
        if isinstance(x, str):
            x = [x]
    
        encodings = self.tokenizer(
            x,
            return_tensors='pt',
            padding=True,
            truncation=True,
        )
        input_ids = encodings['input_ids'].to(self.device)
        attention_mask = encodings["attention_mask"].to(self.device)
    
        output = self(input_ids, attention_mask)
        return torch.sigmoid(output).cpu().numpy()
