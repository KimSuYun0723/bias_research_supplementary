import matplotlib.pyplot as plt
import pandas as pd

# 데이터 정의
categories = [
    "Age", "Disability_status", "Gender_identity", "Nationality",
    "Physical_appearance", "Race_ethnicity", "Race_x_SES", "Race_x_gender",
    "Religion", "SES", "Sexual_orientation"
]

jmbmt_counts = [387, 115, 412, 164, 104, 621, 714, 965, 86, 224, 52]
bert_counts = [238, 92, 259, 275, 63, 465, 809, 696, 103, 443, 49]

# 데이터프레임 생성
df = pd.DataFrame({
    "Category": categories,
    "UnLog": jmbmt_counts,
    "BERT": bert_counts
})

# 막대 그래프 생성 (색상 조정)
plt.figure(figsize=(12, 6))
x = range(len(categories))
bar_width = 0.35

plt.bar(x, df["UnLog"], width=bar_width, label="UnLog", color="blue", alpha=0.7)
plt.bar([i + bar_width for i in x], df["BERT"], width=bar_width, label="BERT", color="orange", alpha=0.7)

# 축 레이블 및 제목
plt.xlabel("Category", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.title("Comparison of Well-Performed Counts: UnLog vs BERT", fontsize=14)
plt.xticks([i + bar_width / 2 for i in x], categories, rotation=45, ha="right")
plt.legend()

# 축 레이블 및 제목
#plt.grid(axis='y', linestyle='--', alpha=0.7)  # y축에만 점선으로 grid 추가

# 그래프 저장
output_path = "/home/nlpgpu7/ellt/suyun/bias_research/bbq_data/final_result/graph.png"  # 원하는 파일 경로 및 이름
plt.tight_layout()
plt.savefig(output_path, dpi=300)  # 이미지 저장
plt.close()  # 그래프 창 닫기
