import torch
import torch.nn as nn
import numpy as np

print("초파리 뇌 기반 반려 AI 신경망 모델을 구축합니다...\n")

# 1. 가상의 초파리 뇌 신경 연결 데이터(시냅스 행렬) 정의하기
# 실제로는 FlyWire에서 가져온 뉴런 간의 연결 가중치(Synapse Weight) 행렬을 사용합니다.
# 여기서는 이해를 돕기 위해 3개의 뉴런(입력, 연동, 출력)이 서로 연결된 상황을 가정합니다.
# 행렬의 값은 뉴런 A가 뉴런 B에 미치는 영향력(가중치)을 뜻합니다.
synapse_weights = torch.tensor([
    [0.0, 1.5, 0.2],  # 뉴런 1의 연결 상태
    [0.5, 0.0, 2.0],  # 뉴런 2의 연결 상태
    [1.0, 0.1, 0.0]   # 뉴런 3의 연결 상태
], dtype=torch.float32)


# 2. 초파리 뇌 신경망 클래스 정의 (PyTorch 활용)
class FlyBrainAI(nn.Module):
    def __init__(self, brain_weights):
        super(FlyBrainAI, self).__init__()
        # 초파리 뇌의 실제 시냅스 연결 가중치를 AI 모델의 가중치(Weight)로 고정합니다.
        self.fc = nn.Linear(3, 3, bias=False)
        self.fc.weight = nn.Parameter(brain_weights)
        
        # 신경세포의 활성화를 조절하는 비선형 함수 (생물학적 뉴런의 발화 특성 모사)
        self.activation = nn.Tanh()

    def forward(self, x):
        # 자극(입력)이 들어왔을 때 뇌 신경망을 거쳐 반응을 출력합니다.
        out = self.fc(x)
        return self.activation(out)


# 3. 모델 생성 및 테스트
ai_brain = FlyBrainAI(synapse_weights)

# 가상의 감각 자극 입력 (예: 외부 환경의 온도, 냄새, 빛 등의 신호)
# 크기가 3인 입력 벡터를 넣어줍니다.
sensory_input = torch.tensor([[1.0, 0.5, -0.2]], dtype=torch.float32)

# AI 뇌 구동
print(f"외부 자극 입력: {sensory_input.tolist()}")
brain_response = ai_brain(sensory_input)

print(f"초파리 뇌 AI의 신경 반응 출력: {brain_response.tolist()}")
print("\n[성공] 초파리 뇌 신경망 구조를 모사한 AI 모델이 정상적으로 작동했습니다!")