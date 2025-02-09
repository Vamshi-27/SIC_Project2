import torch
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
from src.preprocess import MemeDataset
from src.model import VisionLanguageCaptioningModel
import os


def train(model, data_loader, criterion, optimizer, device, num_epochs=10, model_save_path="models/meme_captioning_model.pth"):
    model.to(device)
    model.train()

    for epoch in range(1, num_epochs + 1):
        total_loss = 0
        model.train()

        for batch_idx, (images, input_ids, attention_mask) in enumerate(data_loader):
            images = images.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            # Forward pass
            outputs = model(images, input_ids, attention_mask)
            loss = criterion(outputs.view(-1, outputs.size(-1)), input_ids.view(-1))

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if (batch_idx + 1) % 10 == 0:
                print(f"[Epoch {epoch}/{num_epochs} | Batch {batch_idx + 1}/{len(data_loader)}] Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(data_loader)
        print(f"Epoch {epoch}/{num_epochs} completed. Average Loss: {avg_loss:.4f}")

    # Save the trained model
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"Model successfully saved at: {model_save_path}")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize model, tokenizer, and loss
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = VisionLanguageCaptioningModel()
    criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Load dataset
    dataset = MemeDataset(data_path="data/memes_data.tsv", images_dir="data/images", tokenizer=tokenizer)
    data_loader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Train the model
    train(model, data_loader, criterion, optimizer, device)
