import transformers
from transformers import AutoTokenizer, AutoModel
import torch

print("Transformers:", transformers.__version__)
print("PyTorch:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())
print("CUDA Version:", torch.version.cuda)

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")
print("Transformers loaded successfully!")