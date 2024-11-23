import pandas as pd

def pair_contexts(data):
    """
    질문별로 ambig context와 disambig context를 1:1로 매칭합니다.
    Args:
        data: 원본 데이터프레임 (question, context, context_condition 포함)
    Returns:
        paired_df: 질문별 1:1 매칭된 데이터프레임
    """
    paired_data = []

    # 질문을 기준으로 그룹화
    grouped = data.groupby("question")
    
    for question, group in grouped:
        # ambig와 disambig context 추출
        ambig_contexts = group[group['context_condition'] == 'ambig']['context'].tolist()
        disambig_contexts = group[group['context_condition'] == 'disambig']['context'].tolist()
        
        # ambig와 disambig를 1:1 매칭
        max_len = max(len(ambig_contexts), len(disambig_contexts))
        for i in range(max_len):
            ambig_context = ambig_contexts[i] if i < len(ambig_contexts) else None
            disambig_context = disambig_contexts[i] if i < len(disambig_contexts) else None
            paired_data.append({
                "question": question,
                "ambig_context": ambig_context,
                "disambig_context": disambig_context
            })

    # 새로운 DataFrame으로 변환
    paired_df = pd.DataFrame(paired_data)
    return paired_df

# 데이터 로드
file_path = "/home/nlpgpu7/ellt/suyun/bias_research/bbq_data/preprocessed_bbq/cleaned_bbq_set1.jsonl"
data = pd.read_json(file_path, lines=True)

# ambig-disambig 1:1 매칭
paired_df = pair_contexts(data)

# 결과 저장
output_path = "/home/nlpgpu7/ellt/suyun/bias_research/bbq_data/preprocessed_bbq/organized_bbq_set1.jsonl"
paired_df.to_json(output_path, orient="records", lines=True)

print(f"결과 저장 완료: {output_path}")