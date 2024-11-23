import pandas as pd

# SETTING 1: BERT VS JMBMT (HS)
print("=== SETTING 1: BERT VS JMBMT (HS) ===")
bert_csv_path = "/home/nlpgpu7/ellt/suyun/bias_research/setting1/bert/bert_hs_set1.csv"
our_model_csv_path = "/home/nlpgpu7/ellt/suyun/bias_research/setting1/jmbmt/jmbmt_hs_set1.csv"
bert_df = pd.read_csv(bert_csv_path)
our_model_df = pd.read_csv(our_model_csv_path)

bert_true_count = bert_df['ambig_more_similar'].sum()  # True는 1로 계산
our_model_true_count = our_model_df['ambig_more_similar'].sum()

print(f"BERT 모델에서 True가 나온 개수: {bert_true_count}")
print(f"우리 모델에서 True가 나온 개수: {our_model_true_count}")

if bert_true_count > our_model_true_count:
    print("BERT 모델이 더 많은 True 값을 가짐")
elif bert_true_count < our_model_true_count:
    print("우리 모델이 더 많은 True 값을 가짐")
else:
    print("두 모델이 같은 수의 True 값을 가짐")


# SETTING 1: BERT VS JMBMT (POOLED OUTPUT)
print("=== SETTING 1: BERT VS JMBMT (POOLED OUTPUT) ===")
bert_csv_path = "/home/nlpgpu7/ellt/suyun/bias_research/setting1/bert/bert_pooled_set1.csv"
our_model_csv_path = "/home/nlpgpu7/ellt/suyun/bias_research/setting1/jmbmt/jmbmt_pooled_set1.csv"
bert_df = pd.read_csv(bert_csv_path)
our_model_df = pd.read_csv(our_model_csv_path)

bert_true_count = bert_df['ambig_more_similar'].sum()  # True는 1로 계산
our_model_true_count = our_model_df['ambig_more_similar'].sum()

print(f"BERT 모델에서 True가 나온 개수: {bert_true_count}")
print(f"우리 모델에서 True가 나온 개수: {our_model_true_count}")

if bert_true_count > our_model_true_count:
    print("BERT 모델이 더 많은 True 값을 가짐")
elif bert_true_count < our_model_true_count:
    print("우리 모델이 더 많은 True 값을 가짐")
else:
    print("두 모델이 같은 수의 True 값을 가짐")

###########################################################################
# SETTING 2: BERT VS JMBMT (HS)
print("=== SETTING 2: BERT VS JMBMT (HS) ===")
bert_csv_path = "/home/nlpgpu7/ellt/suyun/bias_research/setting1/bert/bert_hs_set2.csv"
our_model_csv_path = "/home/nlpgpu7/ellt/suyun/bias_research/setting1/jmbmt/jmbmt_hs_set2.csv"
bert_df = pd.read_csv(bert_csv_path)
our_model_df = pd.read_csv(our_model_csv_path)

bert_true_count = bert_df['ambig_more_similar'].sum()  # True는 1로 계산
our_model_true_count = our_model_df['ambig_more_similar'].sum()

print(f"BERT 모델에서 True가 나온 개수: {bert_true_count}")
print(f"우리 모델에서 True가 나온 개수: {our_model_true_count}")

if bert_true_count > our_model_true_count:
    print("BERT 모델이 더 많은 True 값을 가짐")
elif bert_true_count < our_model_true_count:
    print("우리 모델이 더 많은 True 값을 가짐")
else:
    print("두 모델이 같은 수의 True 값을 가짐")


# SETTING 2: BERT VS JMBMT (POOLED OUTPUT)
print("=== SETTING 1: BERT VS JMBMT (POOLED OUTPUT) ===")
bert_csv_path = "/home/nlpgpu7/ellt/suyun/bias_research/setting1/bert/bert_pooled_set2.csv"
our_model_csv_path = "/home/nlpgpu7/ellt/suyun/bias_research/setting1/jmbmt/jmbmt_pooled_set2.csv"
bert_df = pd.read_csv(bert_csv_path)
our_model_df = pd.read_csv(our_model_csv_path)

bert_true_count = bert_df['ambig_more_similar'].sum()  # True는 1로 계산
our_model_true_count = our_model_df['ambig_more_similar'].sum()

print(f"BERT 모델에서 True가 나온 개수: {bert_true_count}")
print(f"우리 모델에서 True가 나온 개수: {our_model_true_count}")

if bert_true_count > our_model_true_count:
    print("BERT 모델이 더 많은 True 값을 가짐")
elif bert_true_count < our_model_true_count:
    print("우리 모델이 더 많은 True 값을 가짐")
else:
    print("두 모델이 같은 수의 True 값을 가짐")