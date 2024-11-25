# bias_research_supplementary
This repository is about a supplementary experiment for JMBMT.

## 1. File Tree
📦bbq_data
 ┣ 📂preprocessed_bbq                    // BBQ 데이터셋 전처리해둔 파일     
 ┃ ┣ 📜cleaned_bbq_set1.jsonl            // Setting1 BBQ 원본 jsonl(question,context,category,polarity..etc)     
 ┃ ┣ 📜final_combined.csv                // 필요한 모든 것이 들어있는 csv !!     
 ┃ ┣ 📜labeled_organized_bbq_set1.csv    // BERT VS JMBMT 중 더 잘한 것 label된 csv     
 ┃ ┣ 📜labeled_organized_bbq_set1.jsonl  // BERT VS JMBMT 중 더 잘한 것 label된 jsonl     
 ┃ ┣ 📜organized_bbq_set1.jsonl          // Setting1 BBQ jsonl(question, ambig, disambig context)     
 ┃ ┣ 📜organized_bbq_set1_2.csv          // Setting2 BBQ csv(question, ambig, disambig context)     
 ┃ ┗ 📜organized_bbq_set2.jsonl          // Setting2 BBQ jsonl(question, ambig, disambig context)     
 ┣ 📜bbq.py                              // BBQ 전처리하기.py     
 ┣ 📜include_ambig.py                    // Setting2로 가공하여 저장하기.py     
 ┗ 📜organize_bbq.py                     // 전처리한 BBQ question 기준으로 organize 하기.py     
     
📦setting1                               // disambig_context가 ambig_context를 포함하고 있는 그대로의 setting     
 ┣ 📂bert     
 ┃ ┣ 📜bert_hs_set1.csv     
 ┃ ┣ 📜bert_hs_set1.py     
 ┃ ┣ 📜bert_pooled_set1.csv     
 ┃ ┗ 📜bert_pooled_set1.py     
 ┗ 📂jmbmt     
 ┃ ┣ 📜jmbmt_hs_set1.csv     
 ┃ ┣ 📜jmbmt_hs_set1.py     
 ┃ ┣ 📜jmbmt_pooled_set1.csv     
 ┃ ┗ 📜jmbmt_pooled_set1.py     
     
 📦setting2                              // disambig_context가 포함하고 있는 ambig_context 제거한 setting     
 ┣ 📂bert     
 ┃ ┣ 📜bert_hs_set2.csv     
 ┃ ┣ 📜bert_hs_set2.py     
 ┃ ┣ 📜bert_pooled_set2.csv     
 ┃ ┗ 📜bert_pooled_set2.py      
 ┗ 📂jmbmt     
 ┃ ┣ 📜jmbmt_hs_set2.csv     
 ┃ ┣ 📜jmbmt_hs_set2.py     
 ┃ ┣ 📜jmbmt_pooled_set2.csv     
 ┃ ┗ 📜jmbmt_pooled_set2.py     
 
📜save_result.py                         // 최종 결과 데이터 저장하기.py(final_combine.csv)     
📜simil_result.py                        // cosine similarity 계산하기.py     
📜sys.py     
📜test.py     