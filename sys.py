import sys

# 모듈 경로를 sys.path에 추가
module_path = "/home/nlpgpu7/ellt/suyun/bias_research/simcse/SimCSE/pretraining"
if module_path not in sys.path:
    sys.path.append(module_path)

# 이제 simcse.py를 import 가능
from simcse import SimCSEPretraining
print("SimCSEPretraining imported successfully!")