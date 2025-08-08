from transformers import RobertaTokenizer, RobertaModel
import torch.optim as optim
from tqdm import tqdm
from torch import nn
import torch


class WazuhCorrelator(nn.Module):
    def __init__(self, bert_model_name: str = "microsoft/codebert-base", hidden_dim: int = 2048, device: str = "cpu"):
        super(WazuhCorrelator, self).__init__()
        self.device = device

        self.tokenizer = RobertaTokenizer.from_pretrained(bert_model_name)
        self.bert = RobertaModel.from_pretrained(bert_model_name)
        for param in self.bert.parameters():
            param.requires_grad = False

        d_model = self.bert.config.hidden_size * 4

        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = None
        self.best_loss = float("inf")

        self.fc = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, inputs, mask):
        def mean_pool(last_hidden_state, attention_mask):
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
            sum_mask = input_mask_expanded.sum(dim=1)
            return sum_embeddings / (sum_mask + 1e-9)

        event_output = self.bert(input_ids=inputs["event_input"], attention_mask=mask["event_mask"]).last_hidden_state
        event_pooled = mean_pool(event_output, mask["event_mask"])

        title_output = self.bert(input_ids=inputs["title_input"], attention_mask=mask["title_mask"]).last_hidden_state
        title_pooled = mean_pool(title_output, mask["title_mask"])

        description_output = self.bert(input_ids=inputs["description_input"], attention_mask=mask["description_mask"]).last_hidden_state
        description_pooled = mean_pool(description_output, mask["description_mask"])

        activity_output = self.bert(input_ids=inputs["activity_input"], attention_mask=mask["activity_mask"]).last_hidden_state
        activity_pooled = mean_pool(activity_output, mask["activity_mask"])

        combined = torch.cat([event_pooled, title_pooled, description_pooled, activity_pooled], dim=1)

        x = self.fc(combined)
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
        text_tuples, labels = zip(*batch)
        events = []
        titles = []
        descriptions = []
        activities = []

        for text_tuple in text_tuples:
            events.append(text_tuple[0])
            titles.append(text_tuple[1])
            descriptions.append(text_tuple[2])
            activities.append(text_tuple[3])

        return events, titles, descriptions, activities, labels

    def train_model(self, batch_size, n_epochs, lr, train_dataset, test_dataset, save_path="model.pt"):
        self.train()

        self.optimizer = optim.AdamW(self.parameters(), lr=lr)
        # self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.5)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, collate_fn=self._collate_fn)
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
            events, titles, descriptions, activities, labels = batch

            event_encodings = self.tokenizer(
                list(events),
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            title_encodings = self.tokenizer(
                list(titles),
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            description_encodings = self.tokenizer(
                list(descriptions),
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            activity_encodings = self.tokenizer(
                list(activities),
                return_tensors="pt",
                padding=True,
                truncation=True,
            )

            inputs = {
                "event_input": event_encodings["input_ids"].to(self.device),
                "title_input": title_encodings["input_ids"].to(self.device),
                "description_input": description_encodings["input_ids"].to(self.device),
                "activity_input": activity_encodings["input_ids"].to(self.device),
            }

            masks = {
                "event_mask": event_encodings["attention_mask"].to(self.device),
                "title_mask": title_encodings["attention_mask"].to(self.device),
                "description_mask": description_encodings["attention_mask"].to(self.device),
                "activity_mask": activity_encodings["attention_mask"].to(self.device),
            }

            labels = torch.tensor(labels, dtype=torch.float, device=self.device)

            self.optimizer.zero_grad()
            outputs = self(inputs, masks)
            loss = self.criterion(outputs.view(-1), labels)
            loss.backward()
            self.optimizer.step()
            # self.scheduler.step()  # Uncomment if using scheduler

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def _validate_epoch(self, val_loader):
        self.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                events, titles, descriptions, activities, labels = batch

                event_encodings = self.tokenizer(
                    list(events),
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                )
                title_encodings = self.tokenizer(
                    list(titles),
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                )
                description_encodings = self.tokenizer(
                    list(descriptions),
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                )
                activity_encodings = self.tokenizer(
                    list(activities),
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                )

                inputs = {
                    "event_input": event_encodings["input_ids"].to(self.device),
                    "title_input": title_encodings["input_ids"].to(self.device),
                    "description_input": description_encodings["input_ids"].to(self.device),
                    "activity_input": activity_encodings["input_ids"].to(self.device),
                }

                masks = {
                    "event_mask": event_encodings["attention_mask"].to(self.device),
                    "title_mask": title_encodings["attention_mask"].to(self.device),
                    "description_mask": description_encodings["attention_mask"].to(self.device),
                    "activity_mask": activity_encodings["attention_mask"].to(self.device),
                }

                labels = torch.tensor(labels, dtype=torch.float, device=self.device)

                outputs = self(inputs, masks)
                loss = self.criterion(outputs.view(-1), labels)

                total_loss += loss.item()

        return total_loss / len(val_loader)

    @torch.no_grad()
    def predict(self, event, title, description, activity):
        self.eval()

        # Ensure all inputs are lists
        if isinstance(event, str):
            event = [event]
        if isinstance(title, str):
            title = [title]
        if isinstance(description, str):
            description = [description]
        if isinstance(activity, str):
            activity = [activity]

        event_encodings = self.tokenizer(
            event,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        title_encodings = self.tokenizer(
            title,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        description_encodings = self.tokenizer(
            description,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        activity_encodings = self.tokenizer(
            activity,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        inputs = {
            "event_input": event_encodings["input_ids"].to(self.device),
            "title_input": title_encodings["input_ids"].to(self.device),
            "description_input": description_encodings["input_ids"].to(self.device),
            "activity_input": activity_encodings["input_ids"].to(self.device),
        }

        masks = {
            "event_mask": event_encodings["attention_mask"].to(self.device),
            "title_mask": title_encodings["attention_mask"].to(self.device),
            "description_mask": description_encodings["attention_mask"].to(self.device),
            "activity_mask": activity_encodings["attention_mask"].to(self.device),
        }

        output = self(inputs, masks)
        return torch.sigmoid(output).cpu().numpy()
