import pandas as pd

# 1. 첫 번째 데이터셋 읽어오기 (jsonl -> csv로 변환)
original_jsonl_path = "/home/nlpgpu7/ellt/suyun/bias_research/bbq_data/preprocessed_bbq/cleaned_bbq_set1.jsonl"
original_df = pd.read_json(original_jsonl_path, lines=True)

# 첫 번째 데이터셋에서 'question'별로 묶기 (ambig_context와 disambig_context가 1:1로 매칭되어야 함)
def pair_contexts(data):
    """
    질문별로 ambig context와 disambig context를 1:1로 매칭합니다.
    """
    paired_data = []
    grouped = data.groupby("question")
    
    for question, group in grouped:
        ambig_contexts = group[group['context_condition'] == 'ambig']['context'].tolist()
        disambig_contexts = group[group['context_condition'] == 'disambig']['context'].tolist()
        
        # ambig와 disambig를 1:1 매칭
        max_len = max(len(ambig_contexts), len(disambig_contexts))
        for i in range(max_len):
            ambig_context = ambig_contexts[i] if i < len(ambig_contexts) else None
            disambig_context = disambig_contexts[i] if i < len(disambig_contexts) else None
            paired_data.append({
                "question": question,
                "category": group['category'].iloc[0],  # category를 1:1로 복사
                "question_polarity": group['question_polarity'].iloc[0],  # question_polarity도 추가
                "ambig_context": ambig_context,
                "disambig_context": disambig_context
            })

    # 새로운 DataFrame으로 변환
    paired_df = pd.DataFrame(paired_data)
    return paired_df

# 'question'별로 ambig/disambig 컨텍스트 매칭
paired_df = pair_contexts(original_df)

# 첫 번째 데이터셋을 CSV로 저장 (기본 4개의 컬럼 포함)
original_csv_path = "/home/nlpgpu7/ellt/suyun/bias_research/bbq_data/preprocessed_bbq/labeled_organized_bbq_set1_2.csv"
paired_df.to_csv(original_csv_path, index=False)

########################################################################
# 2. BERT 데이터셋 읽어오기
bert_csv_path = "/home/nlpgpu7/ellt/suyun/bias_research/setting1/bert/bert_pooled_set1.csv"
bert_df = pd.read_csv(bert_csv_path)

# BERT 데이터에서 필요한 열을 가져와 첫 번째 데이터셋에 추가
paired_df['bert_ambig_simil'] = bert_df['ambig_simil']
paired_df['bert_disambig_simil'] = bert_df['disambig_simil']
paired_df['bert_ambig_more_similar'] = bert_df['ambig_more_similar']

# 3. JMBMT 데이터셋 읽어오기
jmbmt_csv_path = "/home/nlpgpu7/ellt/suyun/bias_research/setting1/jmbmt/jmbmt_pooled_set1.csv"
jmbmt_df = pd.read_csv(jmbmt_csv_path)

# JMBMT 데이터에서 필요한 열을 가져와 첫 번째 데이터셋에 추가
paired_df['jmbmt_ambig_simil'] = jmbmt_df['ambig_simil']
paired_df['jmbmt_disambig_simil'] = jmbmt_df['disambig_simil']
paired_df['jmbmt_ambig_more_similar'] = jmbmt_df['ambig_more_similar']

# 4. 레이블링: 각 행에 대해서 '우리가 잘한 것' (jmbmt_good) 카운트
# 'label'이 포함된 데이터셋 읽어오기
label_jsonl_path = "/home/nlpgpu7/ellt/suyun/bias_research/bbq_data/preprocessed_bbq/labeled_organized_bbq_set1.jsonl"
label_df = pd.read_json(label_jsonl_path, lines=True)

# 'label' 열을 paired_df에 추가 (같은 'question' 기준으로 병합)
paired_df['label'] = label_df['label']

# 5. 카테고리별로 '우리가 잘한 것' (jmbmt_good) 개수 세기
category_counts = paired_df[paired_df['label'] == 'jmbmt_good'].groupby('category').size()

# 6. 결과 출력
print("카테고리별 '우리가 잘한 것' 개수:")
print(category_counts)

# 7. 최종 데이터셋을 CSV로 저장
final_csv_path = "/home/nlpgpu7/ellt/suyun/bias_research/bbq_data/preprocessed_bbq/final_combined.csv"
paired_df.to_csv(final_csv_path, index=False)

print(f"최종 데이터셋이 저장되었습니다: {final_csv_path}")
