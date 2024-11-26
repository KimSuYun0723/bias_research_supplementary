import pandas as pd

# SETTING 1: BERT VS JMBMT (POOLED OUTPUT)
print("\n=== SETTING 1: BERT VS JMBMT (POOLED OUTPUT) ===")
bert_csv_path = "/home/nlpgpu7/ellt/suyun/bias_research/setting1/bert_pooled_set1.csv"
our_model_csv_path = "/home/nlpgpu7/ellt/suyun/bias_research/setting1/jmbmt_pooled_set1.csv"
bert_df = pd.read_csv(bert_csv_path)
our_model_df = pd.read_csv(our_model_csv_path)

################################################################################
# BERT에서 True, JMBMT에서 False인 경우
both_true = (bert_df['ambig_more_similar'] == True) & (our_model_df['ambig_more_similar'] == True)
both_true_count = both_true.sum()
bert_true_jmbmt_false = (bert_df['ambig_more_similar'] == True) & (our_model_df['ambig_more_similar'] == False)
bert_true_jmbmt_false_count = bert_true_jmbmt_false.sum()

# BERT에서 False, JMBMT에서 True인 경우
both_false = (bert_df['ambig_more_similar'] == False) & (our_model_df['ambig_more_similar'] == False)
both_false_count = both_false.sum()
bert_false_jmbmt_true = (bert_df['ambig_more_similar'] == False) & (our_model_df['ambig_more_similar'] == True)
bert_false_jmbmt_true_count = bert_false_jmbmt_true.sum()

# 레벨별로 분류된 표 출력
comparison_table = pd.DataFrame(
    {
        "Performance": ["둘다 못함", "우리가 더 잘함", "BERT가 더 잘함", "둘다 잘함"],
        "Num": [both_true_count, bert_true_jmbmt_false_count, bert_false_jmbmt_true_count, both_false_count]
    },
    columns=["Performance", "Num"]
)
print("\n=== 결과 표 ===")
print(comparison_table)

# 레이블링된 데이터를 원본 데이터셋에 추가
original_path = "/home/nlpgpu7/ellt/suyun/bias_research/bbq_data/preprocessed_bbq/organized_bbq_set1.jsonl"
original_df = pd.read_json(original_path, lines=True)

def label_row(bert_row, jmbmt_row):
    if (bert_row['ambig_more_similar'] == True) and (jmbmt_row['ambig_more_similar'] == True):
        return "both_bad"
    elif (bert_row['ambig_more_similar'] == False) and (jmbmt_row['ambig_more_similar'] == False):
        return "both_good"
    elif (bert_row['ambig_more_similar'] == True) and (jmbmt_row['ambig_more_similar'] == False):
        return "jmbmt_good"
    elif (bert_row['ambig_more_similar'] == False) and (jmbmt_row['ambig_more_similar'] == True):
        return "bert_good"

# 레이블링된 데이터셋 저장
labels = []
for i, (bert_row, jmbmt_row) in enumerate(zip(bert_df.iterrows(), our_model_df.iterrows())):
    labels.append(label_row(bert_row[1], jmbmt_row[1]))  # 행을 비교해서 레이블링

original_df['label'] = labels  # 원본 데이터셋에 레이블 추가

# 새로 레이블링된 데이터셋 저장
output_path = "/home/nlpgpu7/ellt/suyun/bias_research/bbq_data/result_data/labeled_organized_bbq_set1.jsonl"
original_df.to_json(output_path, orient="records", lines=True)

print(f"새로운 레이블이 추가된 데이터셋을 저장했습니다: {output_path}")

print("\n=== 마지막 확인 ===")
print(f"둘다 True(=둘다 못함): {original_df[original_df['label'] == 'both_bad'].shape[0]}")
print(f"BERT True & JMBMT False(=우리가 더 잘함): {original_df[original_df['label'] == 'jmbmt_good'].shape[0]}")
print(f"BERT False & JMBMT True(=BERT가 더 잘함): {original_df[original_df['label'] == 'bert_good'].shape[0]}")
print(f"둘다 False(=둘다 잘함): {original_df[original_df['label'] == 'both_good'].shape[0]}")

"""

"""