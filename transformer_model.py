"""
transformer-based packet classifier

run python transformer_model.py to execute model
"""

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from torch.optim import AdamW


CSV_PATH = "packets_dataset.csv"
MODEL_NAME = "distilbert-base-uncased"  # you can change this
MAX_LENGTH = 64
BATCH_SIZE = 16
EPOCHS = 2
LR = 2e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------ DATASET ------------------ #
def build_text(row: pd.Series) -> str:
    """
    Turn one packet row into a text 'sentence' the transformer can read.

    Supports both:
    - Older CSV schema: timestamp, ip_src, ip_dst, mac_src, mac_dst, protocols, length
    - Newer AiNDS.py schema: Timestamp, IP Source, IP Destination, MAC Source, MAC Destination, Protocol, Length
    """

    def get_val(series: pd.Series, *keys, default: str = "0") -> str:
        for key in keys:
            if key in series.index:
                val = series[key]
                return "0" if pd.isna(val) else str(val)
        return default

    src_ip = get_val(row, "IP Source", "ip_src")
    dst_ip = get_val(row, "IP Destination", "ip_dst")
    src_mac = get_val(row, "MAC Source", "mac_src")
    dst_mac = get_val(row, "MAC Destination", "mac_dst")
    proto = get_val(row, "Protocol", "protocols", default="Unknown")
    length = get_val(row, "Length", "length", default="0")

    return (
        f"src_ip {src_ip} "
        f"dst_ip {dst_ip} "
        f"src_mac {src_mac} "
        f"dst_mac {dst_mac} "
        f"proto {proto} "
        f"length {length}"
    )


class PacketDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 64):
        self.texts = [build_text(row) for _, row in df.iterrows()]
        self.labels = df["Label"].astype(int).tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        text = self.texts[idx]
        label = self.labels[idx]

        enc = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(label, dtype=torch.long)
        return item


# ------------------ TRAIN LOOP ------------------ #
def train_epoch(model, dataloader, optimizer):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}

        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(len(dataloader), 1)


# ------------------ MAIN ------------------ #
def main():
    # 1. Load CSV (produced by capture_packets)
    df = pd.read_csv(CSV_PATH)

    if df.empty:
        print("[TRANSFORMER] CSV is empty. Run AiNDS.capture_packets first.")
        return

    # Ensure a Label column exists. If this CSV is unlabeled,
    # treat everything as benign (0) for now.
    if "Label" not in df.columns:
        df["Label"] = 0

    # TODO: For now, Label is likely all 0 (benign). When you have a
    # labeled dataset (NSL-KDD / CICIDS2017), load/merge labels here.
    print(f"[TRANSFORMER] Loaded {len(df)} rows from {CSV_PATH}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    dataset = PacketDataset(df, tokenizer, max_length=MAX_LENGTH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,  # benign vs malicious
    )
    model.to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=LR)

    for epoch in range(1, EPOCHS + 1):
        avg_loss = train_epoch(model, dataloader, optimizer)
        print(f"[TRANSFORMER] Epoch {epoch}/{EPOCHS} - loss: {avg_loss:.4f}")

    # Save the fine-tuned model
    save_path = "packet_transformer_model"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"[TRANSFORMER] Saved model to {save_path}")


if __name__ == "__main__":
    main()