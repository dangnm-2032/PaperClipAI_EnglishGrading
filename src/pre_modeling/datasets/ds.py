import torch  #  Essential library for numerical computations and deep learning 
from torch.utils.data import Dataset  # ️ Foundation for loading and managing datasets ️
    
# ✨✨✨ DatasetLoader class: Streamlines data preparation for model training ✨✨✨
class DatasetLoader(Dataset):
    def __init__(self, df, mode, cfg):
        self.df = df.reset_index(drop=True)  # Realign data for consistency 
        self.mode = mode  # Specify operational mode (train, validation, or test) ⚙️
        self.cfg = cfg  # Access configuration settings for flexibility ️

        self.tokenizer = cfg.tokenizer  # Load the trusty text tokenizer 
        if self.tokenizer.pad_token is None:  # Handle potential token variations 
            self.tokenizer.pad_token = self.tokenizer.eos_token  # Set padding for coherence 

        self.texts = self.df[cfg.dataset.text_column].values  # Extract text data for fueling the model 
        self.targets = self.df[cfg.dataset.target_cols].values  # Gather target values for guidance 
        self.cfg._tokenizer_cls_token_id = self.tokenizer.cls_token_id  # Store special token IDs for reference ️
        self.cfg._tokenizer_sep_token_id = self.tokenizer.sep_token_id  
        self.cfg._tokenizer_mask_token_id = self.tokenizer.mask_token_id  

    def __len__(self):  # Reveal the dataset's size for planning 
        return len(self.df)  # Transparency is key 
    
    def __getitem__(self, idx):  # Fetch a specific data sample for processing 
        sample = dict()  # Prepare an empty container for our ingredients 
        sample = self._read_data(idx=idx, sample=sample)  # Gather text data 
        if self.targets is not None:  # Include targets if available 
            sample = self._read_label(idx=idx, sample=sample)  # Attach the guiding labels 

        return sample  # Present the neatly packaged sample 
    
    def _read_data(self, idx, sample):  # Focus on text extraction and encoding 
        text = self.texts[idx][0]  # Isolate the text at the designated index 
        sample.update(self.encode(text))  # Incorporate encoded text into the sample 
        return sample  # Return the enriched sample ✨
    
    def _read_label(self, idx, sample):  # Handle label retrieval with care 
        sample["target"] = torch.tensor(self.targets[idx], dtype=torch.float)  # Convert label to a numerical torch tensor for compatibility 
        return sample  # Return the sample enhanced with the label 

    def encode(self, text):  # Masterfully transform text into numerical representations ‍♂️
        sample = dict()  # Prepare a fresh dictionary for encoded elements ️
        encodings = self.tokenizer.encode_plus(
            text,  # Feed the text to the tokenizer's magic portal ✨
            return_tensors=None,  # Opt for direct numerical arrays 
            add_special_tokens=True,  # Include those significant special tokens 
            max_length=self.cfg.dataset.max_len,  # Set a boundary for consistency ✂️
            pad_to_max_length=True,  # Ensure uniform length for harmony 
            truncation=True  # Gracefully handle text that exceeds limits 
        )

        sample["input_ids"] = torch.tensor(encodings["input_ids"], dtype=torch.long)  # Store token IDs as a torch tensor for calculations 
        sample["attention_mask"] = torch.tensor(encodings["attention_mask"], dtype=torch.long)  # Include the attention mask for focus guidance 
        return sample  # Deliver the encoded masterpiece 