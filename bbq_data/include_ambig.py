import pandas as pd

def pair_contexts_and_remove_ambig(data):
    """
    질문별로 ambig context와 disambig context를 1:1로 매칭하고,
    disambig_context에서 ambig_context를 제거한 결과를 추가.
    Args:
        data: 원본 데이터프레임 (question, context, context_condition, category 포함)
    Returns:
        paired_df: 질문별 1:1 매칭된 데이터프레임 (disambig_minus_ambig 추가)
    """
    paired_data = []

    # 질문을 기준으로 그룹화
    grouped = data.groupby("question")
    
    for question, group in grouped:
        # category 추출 (질문당 하나의 category만 있다고 가정)
        category = group['category'].iloc[0]
        
        # ambig와 disambig context 추출
        ambig_contexts = group[group['context_condition'] == 'ambig']['context'].tolist()
        disambig_contexts = group[group['context_condition'] == 'disambig']['context'].tolist()
        
        # ambig와 disambig를 1:1 매칭
        max_len = max(len(ambig_contexts), len(disambig_contexts))
        for i in range(max_len):
            ambig_context = ambig_contexts[i] if i < len(ambig_contexts) else None
            disambig_context = disambig_contexts[i] if i < len(disambig_contexts) else None
            
            # disambig_context에서 ambig_context 제거
            disambig_minus_ambig = None
            if ambig_context and disambig_context and ambig_context in disambig_context:
                disambig_minus_ambig = disambig_context.replace(ambig_context, "").strip()
            
            # 디버깅 출력
            print(f"Category: {category}")
            print(f"Question: {question}")
            print(f"Ambig Context: {ambig_context}")
            print(f"Disambig Context: {disambig_context}")
            print(f"Disambig Minus Ambig: {disambig_minus_ambig}")
            print("-" * 50)
            
            paired_data.append({
                "category": category,
                "question": question,
                "ambig_context": ambig_context,
                "disambig_context": disambig_context,
                "disambig_minus_ambig": disambig_minus_ambig
            })

    # 새로운 DataFrame으로 변환
    paired_df = pd.DataFrame(paired_data)
    return paired_df


# 데이터 로드
file_path = "/home/nlpgpu7/ellt/suyun/bias_research/bbq_data/preprocessed_bbq/cleaned_bbq_set1.jsonl"
data = pd.read_json(file_path, lines=True)

# ambig-disambig 1:1 매칭 및 ambig 제거
paired_df = pair_contexts_and_remove_ambig(data)

# 결과 저장
output_path = "/home/nlpgpu7/ellt/suyun/bias_research/bbq_data/preprocessed_bbq/organized_bbq_disambig_minus_ambig.jsonl"
paired_df.to_json(output_path, orient="records", lines=True)

print(f"결과 저장 완료: {output_path}")


"""def pair_contexts_with_labels_and_category(data):
    paired_data = []

    # 질문을 기준으로 그룹화
    grouped = data.groupby("question")
    
    for question, group in grouped:
        # category 추출 (질문당 하나의 category만 있다고 가정)
        category = group['category'].iloc[0]
        
        # ambig와 disambig context 추출
        ambig_contexts = group[group['context_condition'] == 'ambig']['context'].tolist()
        disambig_contexts = group[group['context_condition'] == 'disambig']['context'].tolist()
        
        # ambig와 disambig를 1:1 매칭
        max_len = max(len(ambig_contexts), len(disambig_contexts))
        for i in range(max_len):
            ambig_context = ambig_contexts[i] if i < len(ambig_contexts) else None
            disambig_context = disambig_contexts[i] if i < len(disambig_contexts) else None
            
            # Label 계산: 정확히 포함되는지 확인
            label = 1 if ambig_context and disambig_context and ambig_context in disambig_context else 0
            
            # 디버깅 출력
            print(f"Category: {category}")
            print(f"Question: {question}")
            print(f"Ambig Context: {ambig_context}")
            print(f"Disambig Context: {disambig_context}")
            print(f"Label: {label}")
            print("-" * 50)
            
            paired_data.append({
                "category": category,
                "question": question,
                "ambig_context": ambig_context,
                "disambig_context": disambig_context,
                "label": label
            })

    # 새로운 DataFrame으로 변환
    paired_df = pd.DataFrame(paired_data)
    return paired_df


# 데이터 로드
file_path = "/home/nlpgpu7/ellt/suyun/bias_research/bbq_data/preprocessed_bbq/cleaned_bbq_set1.jsonl"
data = pd.read_json(file_path, lines=True)

# ambig-disambig 1:1 매칭 및 레이블 추가
paired_df = pair_contexts_with_labels_and_category(data)

# 레이블 개수 확인
label_counts = paired_df['label'].value_counts()
print(f"Label이 1인 데이터 개수: {label_counts.get(1, 0)}")
print(f"Label이 0인 데이터 개수: {label_counts.get(0, 0)}")

# 결과 저장
output_path = "/home/nlpgpu7/ellt/suyun/bias_research/bbq_data/preprocessed_bbq/organized_bbq_with_labels_and_category.jsonl"
paired_df.to_json(output_path, orient="records", lines=True)

print(f"결과 저장 완료: {output_path}")"""