import pandas as pd


# 데이터 로드 (위에서 이미 CSV로 변환된 데이터)
final_df_path = "/home/nlpgpu7/ellt/suyun/bias_research/bbq_data/final_result/final_combined.csv"
final_df = pd.read_csv(final_df_path)

# JMBMT가 가장 잘한 것 저장

"""# 1. jmbmt_good인 데이터만 필터링(이미 jmbmt는 disambig>ambig, bert는 ambig>disambig)
jmbmt_good_df = final_df[final_df['label'] == 'both_bad']
print(f"1. jmbmt_good인 데이터만 필터링: {len(jmbmt_good_df)}")

# 2. JMBMT의 disambig - ambig 차이 VS bert의 ambig - disambig 차이
jmbmt_good_df = jmbmt_good_df[(jmbmt_good_df['jmbmt_disambig_simil'] - jmbmt_good_df['jmbmt_ambig_simil']) >= 0.1] # 우리가 더 못한 것은 없다
print(f"2. JMBMT의 ambig - disambig 차이가 클 것: {len(jmbmt_good_df)}")

# 3. BERT의 ambig - disambig 차이가 클 것 (= bias 있는걸 더 잘 짚어내므로 잘못됨)
jmbmt_good_df = jmbmt_good_df[(jmbmt_good_df['bert_ambig_simil'] - jmbmt_good_df['bert_disambig_simil']) >= 0.1]
print(f"3. BERT의 ambig - disambig 차이가 적을 것: {len(jmbmt_good_df)}")"""

#++
jmbmt_good_df = final_df[(final_df['bert_ambig_simil'] < final_df['jmbmt_ambig_simil'])]
print(f"BERT보다 JMBMT가 bias를 더 잡아낸다: {len(jmbmt_good_df)}")

"""
# 4. BERT가 q~ambig 유사도를 높게 예측했어야 한다 (반면 틀렸음)
# 예를 들어, bert_ambig_simil 값이 0.7 이상인 것 (threshold 설정)
jmbmt_good_df = jmbmt_good_df[jmbmt_good_df['bert_ambig_simil'] > 0.7]
print(f"4. BERT가 q~ambig 유사도를 높게: {len(jmbmt_good_df)}")

# 5. JMBMT는 q~ambig 유사도를 낮게 예측해야 함 (반면 틀림)
# 예를 들어, jmbmt_ambig_simil 값이 0.3 이하인 것 (threshold 설정)
jmbmt_good_df = jmbmt_good_df[jmbmt_good_df['jmbmt_ambig_simil'] < 0.3]

# 결과 출력
print(f"조건을 만족하는 'jmbmt_good' 데이터 수: {len(jmbmt_good_df)}")"""

# 6. 결과를 CSV로 저장
filtered_csv_path = "/home/nlpgpu7/ellt/suyun/bias_research/bbq_data/final_result/filtered_bert_good.csv"
jmbmt_good_df.to_csv(filtered_csv_path, index=False)
print(f"조건을 만족하는 데이터가 저장되었습니다: {filtered_csv_path}")

"""# 평균 내기
overall_averages = {
    'jmbmt_ambig_simil_avg': final_df['jmbmt_ambig_simil'].mean(),
    'jmbmt_disambig_simil_avg': final_df['jmbmt_disambig_simil'].mean(),
    'bert_ambig_simil_avg': final_df['bert_ambig_simil'].mean(),
    'bert_disambig_simil_avg': final_df['bert_disambig_simil'].mean()
}

# 4개 레이블 별로 평균 계산
label_averages = final_df.groupby('label').agg({
    'jmbmt_ambig_simil': 'mean',
    'jmbmt_disambig_simil': 'mean',
    'bert_ambig_simil': 'mean',
    'bert_disambig_simil': 'mean'
})

# 결과 출력
print("전체 평균 값:")
for key, value in overall_averages.items():
    print(f"{key}: {value}")

print("\n4개 레이블 별 평균 값:")
print(label_averages)

# 6. 결과를 CSV로 저장
label_averages_csv_path = "/home/nlpgpu7/ellt/suyun/bias_research/bbq_data/final_result/label_averages.csv"
label_averages.to_csv(label_averages_csv_path)
print(f"4개 레이블 별 평균이 저장되었습니다: {label_averages_csv_path}")"""