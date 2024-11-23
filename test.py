# transformers & torch test
"""import transformers
from transformers import AutoTokenizer, AutoModel
import torch

print("Transformers:", transformers.__version__)
print("PyTorch:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())
print("CUDA Version:", torch.version.cuda)

tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")
model = AutoModel.from_pretrained("bert-large-uncased")
print("Transformers loaded successfully!")"""

# rouge test
from rouge_score import rouge_scorer

def test_rouge_scorer():
    """
    rouge_scorer 테스트 함수
    """
    # 테스트 데이터
    question = "What is the capital of France?"
    context = "The capital of France is Paris."

    # ROUGE-L 계산
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(question, context)

    # 결과 출력
    print("ROUGE-L Precision:", scores['rougeL'].precision)
    print("ROUGE-L Recall:", scores['rougeL'].recall)
    print("ROUGE-L F1:", scores['rougeL'].fmeasure)

# 테스트 실행
test_rouge_scorer()