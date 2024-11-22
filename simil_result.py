import pandas as pd

# CSV 파일 경로 (예시)
bert_csv_path = "/home/nlpgpu7/ellt/suyun/bias_research/bert/bert_simil_results.csv"
our_model_csv_path = "/home/nlpgpu7/ellt/suyun/bias_research/jmbmt/jmbmt_simil_results.csv"

# CSV 파일 로드
bert_df = pd.read_csv(bert_csv_path)
our_model_df = pd.read_csv(our_model_csv_path)

# 'ambig_more_similar' 열에서 True 값의 개수를 셈
bert_true_count = bert_df['ambig_more_similar'].sum()  # True는 1로 계산
our_model_true_count = our_model_df['ambig_more_similar'].sum()

# 출력
print(f"BERT 모델에서 True가 나온 개수: {bert_true_count}")
print(f"우리 모델에서 True가 나온 개수: {our_model_true_count}")

# 모델 비교
if bert_true_count > our_model_true_count:
    print("BERT 모델이 더 많은 True 값을 가짐")
elif bert_true_count < our_model_true_count:
    print("우리 모델이 더 많은 True 값을 가짐")
else:
    print("두 모델이 같은 수의 True 값을 가짐")
