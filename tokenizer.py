from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

from datasets import load_dataset

tokenizer = Tokenizer(BPE())
trainer = BpeTrainer(vocab_size=16384)
tokenizer.pre_tokenizer = Whitespace()

ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train+test+validation")

def batch_iterator(batch_size=1000):
    # Only keep the text column to avoid decoding the rest of the columns unnecessarily
    tok_dataset = ds.select_columns("text")
    for batch in tok_dataset.iter(batch_size):
        yield batch["text"]
        
tokenizer.train_from_iterator(batch_iterator(), trainer=trainer, length=len(ds))

tokenizer.save("data/tokenizer-wiki.json")