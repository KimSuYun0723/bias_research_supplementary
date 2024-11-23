from datasets import load_dataset

def select_needed_columns(dataframe, selected_col):
    """
    필수 컬럼만 선택
    """
    if set(selected_col).issubset(dataframe.columns):
        return dataframe[selected_col]
    else: # 선택한 column이 없을 때
        print("Warning: Some columns are missing in the dataset.")
        missing_cols = set(selected_col) - set(dataframe.columns)
        print(f"Missing columns: {missing_cols}") 
    return dataframe # 원본 데이터 반환

def attach_disambig(dataframe, ambig_context_map):
    """
    'disambig'의 context에서 'ambig' context를 제거한 새 문장을 추가
    """
    def process_row(row):
        if row['context_condition'] == 'disambig':
            ambig_context = ambig_context_map[row['question']]
            print("AMBIG:", ambig_context)
            disambig_minus_ambig = row['context'].replace(ambig_context, "").strip()
            print("D - A:", disambig_minus_ambig)
            print()
            # ambig context 제거
            return disambig_minus_ambig
        return None  # 'disambig'이 아닌 경우

    dataframe['disambig_minus_ambig'] = dataframe.apply(process_row, axis=1)
    return dataframe

def save_bbq_dataset(dataset_name, save_path, selected_col, setting=1):
    """
    BBQ 데이터셋 처리 및 저장

    Args:
        dataset_name (str) : "seyoungsong/BBQ"
        save_path (str) : path final dataset(csv) will be saved
        selected_col (list) : list of columns
        setting (int) : 1 - basic processing, 2 - adding disambig context
    """
    print(f"=== LOADING BBQ DATASET... ===")
    dataset = load_dataset(dataset_name, trust_remote_code=True)
    df = dataset['train'].to_pandas()
    negative_df = df[df["question_polarity"] == "neg"] # negative polarity만 추출한 df

    print(f"=== SELECTING COLUMNS... ===")
    wanted_df = select_needed_columns(negative_df, selected_col)
    
    if setting==2:
        print(f"=== SETTING 2: ATTACHING COLUMN... ===")
        # question : context
        ambig_context_map = negative_df[(negative_df['context_condition']=='ambig') & (negative_df['category'] == 'Age')][:10]\
            .set_index('question')['context'].to_dict()
        wanted_df = attach_disambig(wanted_df, ambig_context_map)

    print(f"=== SAVING INTO JSONL... ===")
    #wanted_df.to_csv(save_path, index=False)
    wanted_df.to_json(save_path, orient="records", lines=True, force_ascii=False)

    print(f"=== SUCCESS! ===")
    return wanted_df

# Setting 1
save_bbq_dataset(
    dataset_name= "seyoungsong/BBQ",
    save_path= "/home/nlpgpu7/ellt/suyun/bias_research/bbq_data/preprocessed_bbq/cleaned_bbq_set1.jsonl",
    selected_col=["question_polarity", "category", "question", "context", "context_condition"]
)

# Setting 2
"""save_bbq_dataset(
    dataset_name= "seyoungsong/BBQ",
    save_path= "/home/nlpgpu7/ellt/suyun/bias_research/bbq_data/preprocessed_bbq/cleaned_bbq_set2.jsonl",
    selected_col=["question_polarity", "category", "question", "context", "context_condition"],
    setting=2
)"""

"""
품고 있는 애 VS 안품고 있는 애 개수는 나중에 확인
"""