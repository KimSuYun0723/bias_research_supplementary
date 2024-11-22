import torch
from transformers import AutoTokenizer, AutoModel
from torch.nn.functional import cosine_similarity
import pandas as pd


# Step 1: Data Load
def load_data(file_path):
    """JSONL 파일에서 데이터를 로드합니다."""
    df = pd.read_json(file_path, lines=True)
    return df

# Step 2: BERT Model Load
def load_bert_model(model_name="bert-base-uncased", device='cpu'):
    """BERT 모델과 토크나이저 로드"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    return tokenizer, model

# Step 3: Hidden State
def get_hidden_state(tokenizer, model, text, device="cpu"):
    """CLS hidden state 추출"""
    if not text:
        raise ValueError("Text is None or empty!")
    
    model.eval()

    # Tokenization
    inputs = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt"
    ).to(device)

    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)
        cls_hidden_state = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
        
        if len(text) < 100:  # 텍스트 길이에 따라 출력 제한
            print(f"CLS Hidden State for text: {text}\n{cls_hidden_state}\n")

    return cls_hidden_state 

# Step 4: Caluate Similarity
def calcul_simil(hidden_q, hidden_c):
    """Cosine 유사도 계산"""
    return cosine_similarity(hidden_q, hidden_c, dim=1).item()

# Step 5: Perform TasksW
def perform_task(data, tokenizer, model, device="cpu"):
    """
    질문, ambig context, disambig context로 유사도 계산.
    Args:
        data: DataFrame (question, ambig_context, disambig_context 포함)
        tokenizer: BERT 토크나이저
        model: BERT 모델
        device: 'cuda' or 'cpu'
    Returns:
        결과 DataFrame
    """
    results = []
    error_log = []

    for idx, row in data.iterrows():
        question = row['question']
        ambig_context = row['ambig_context']
        disambig_context = row['disambig_context']

        try:
            # Hidden state 추출
            hidden_q = get_hidden_state(tokenizer, model, question, device)
            hidden_ambig = get_hidden_state(tokenizer, model, ambig_context, device)
            hidden_disambig = get_hidden_state(tokenizer, model, disambig_context, device)
            
            # 유사도 계산
            simil_ambig = calcul_simil(hidden_q, hidden_ambig)
            simil_disambig = calcul_simil(hidden_q, hidden_disambig)

            # 결과 저장
            results.append({
                'question': question,
                'ambig_context': ambig_context,
                'disambig_context': disambig_context,
                'ambig_simil': simil_ambig,
                'disambig_simil': simil_disambig,
                'ambig_more_similar': simil_ambig > simil_disambig
            })
        
        except Exception as e:
            error_log.append((idx, str(e)))
            print(f"Error processing row {idx}: {e}")
            continue
    
    if error_log:
        print(f"총 {len(error_log)}개의 행에서 에러 발생")

    return pd.DataFrame(results)



# Main 
if __name__ == "__main__":
    file_path = "/home/nlpgpu7/ellt/suyun/bias_research/bbq_data/preprocessed_bbq/organized_bbq_set1.jsonl"
    
    data = load_data(file_path)
    print(f"데이터 로드 완료: {len(data)} rows")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"사용 가능한 디바이스: {device}")

    tokenizer, model = load_bert_model("bert-base-uncased", device=device)

    print("=== TASK 시작 ===")
    result_df = perform_task(data, tokenizer, model, device=device)
    print("=== TASK 완료 ===")

    output_path = "/home/nlpgpu7/ellt/suyun/bias_research/bert/bert_simil_results.csv"
    result_df.to_csv(output_path, index=False)
    print(f"결과 저장 완료: {output_path}")