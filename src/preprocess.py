import os
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd

class MemeDataset(Dataset):
    def __init__(self, data_path, images_dir, tokenizer, max_seq_length=50, transform=None):
        self.data_path = data_path
        self.images_dir = images_dir
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.image_caption_pairs = self._load_data()

    def _load_data(self):
        data = pd.read_csv(self.data_path, sep="\t")
        image_caption_pairs = []
        for _, row in data.iterrows():
            image_name = f"{row['HashId']}.jpg"
            image_path = os.path.join(self.images_dir, image_name)
            caption = row["Title"]
            if os.path.exists(image_path) and isinstance(caption, str):
                image_caption_pairs.append((image_path, caption))
        print(f"Loaded {len(image_caption_pairs)} image-caption pairs.")
        return image_caption_pairs

    def __len__(self):
        return len(self.image_caption_pairs)

    def __getitem__(self, idx):
        image_path, caption = self.image_caption_pairs[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Tokenize caption
        tokens = self.tokenizer(caption, padding='max_length', truncation=True, max_length=self.max_seq_length, return_tensors="pt")
        return image, tokens["input_ids"].squeeze(), tokens["attention_mask"].squeeze()
