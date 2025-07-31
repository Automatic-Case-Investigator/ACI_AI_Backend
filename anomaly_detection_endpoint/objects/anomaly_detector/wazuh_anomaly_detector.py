from transformers import GPT2Tokenizer
import torch.optim as optim
from tqdm import tqdm
from torch import nn
import torch
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class WazuhAnomalyDetector(nn.Module):
    def __init__(
        self, d_model: int = 128, dim_feedforward: int = 2048, device: str = "cpu"
    ):
        super(WazuhAnomalyDetector, self).__init__()
        self.d_model = d_model
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.embedding = nn.Embedding(self.tokenizer.vocab_size, self.d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length=512)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=8, dim_feedforward=dim_feedforward, dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6)
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = None
        self.device = device
        self.best_loss = float("inf")

        self.fc = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, 1),
        )

    def forward(self, x, attention_mask=None):
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        x = self.encoder(x, src_key_padding_mask=~attention_mask.bool() if attention_mask is not None else None)
        x = x.mean(dim=1)
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
                max_length=512,
            )
            input_ids = encodings["input_ids"].to(self.device)
            attention_mask = encodings["attention_mask"].to(self.device)
            labels = torch.tensor(labels, dtype=torch.float, device=self.device)
    
            self.optimizer.zero_grad()
            outputs = self(input_ids, attention_mask)
            loss = self.criterion(outputs.view(-1), labels.float())
            loss.backward()
            self.optimizer.step()
    
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
                    max_length=512,
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
            max_length=512
        )
        input_ids = encodings['input_ids'].to(self.device)
        attention_mask = encodings["attention_mask"].to(self.device)
    
        output = self(input_ids, attention_mask)
        return torch.sigmoid(output).cpu().numpy()