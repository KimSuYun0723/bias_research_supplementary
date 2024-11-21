# datasets pandas
import pandas as pd
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
    dataframe['disambig_minus_ambig'] = dataframe.apply(
        lambda row: (
            row['context'].replace(ambig_context_map.get(row['question'],""),"").strip()
            if row['context_condition'] == 'disambig'
            else None
        ),
        axis=1
    )
    return dataframe

def save_bbq_dataset(dataset_name, split, save_path, selected_col, setting=1):
    """
    BBQ 데이터셋 처리 및 저장

    Args:
        dataset_name (str) : "heegyu/bbq"
        split (str) : standard dataset should be divided
        save_path (str) : path final dataset(csv) will be saved
        selected_col (list) : list of columns
        setting (int) : 1 - basic processing, 2 - adding disambig context
    """
    print(f"=== LOADING BBQ DATASET... ===")
    dataset = load_dataset(dataset_name, split=split, trust_remote_code=True)
    df = pd.DataFrame(dataset)
    negative_df = df[df["question_polarity"] == "negative"] # negative polarity만 추출한 df

    print(f"=== SELECTING COLUMNS... ===")
    wanted_df = select_needed_columns(negative_df, selected_col)
    
    if setting==2:
        print(f"=== SETTING 2: ATTACHING COLUMN... ===")
        ambig_context_map = negative_df[negative_df['context_condition']=='ambig'].set_index('question')['context'].to_dict()
        wanted_df = attach_disambig(wanted_df, ambig_context_map)

    print(f"=== SAVING INTO CSV... ===")
    #wanted_df.to_csv(save_path, index=False)
    wanted_df.to_json(save_path, orient="records", lines=True)

    print(f"=== SUCCESS! ===")
    return wanted_df

# Setting 1
data_name = "heegyu/bbq"
save_path = "C:/_SY/bias_research/bbq_data/preprocessed_bbq/cleaned_bbq_set1.json"
selected_columns = ["question_polarity", "categoty", "question", "context", "context_condition", ]

save_bbq_dataset(
    dataset_name=data_name,
    split="test",
    save_path=save_path,
    selected_col=selected_columns
)

# Setting 2
data_name = "heegyu/bbq"
save_path = "C:/_SY/bias_research/bbq_data/preprocessed_bbq/cleaned_bbq_set2.json"
selected_columns = ["question_polarity", "categoty", "question", "context", "context_condition", ]

save_bbq_dataset(
    dataset_name=data_name,
    split="test",
    save_path=save_path,
    selected_col=selected_columns,
    setting=2  # setting==2로 설정하여 새 열 추가
)