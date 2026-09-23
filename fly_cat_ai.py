import torch
import torch.nn as nn
import pygame
import random
import sys

# --- 1. 앞서 구현한 초파리 뇌 AI 모델 정의 ---
class FlyBrainAI(nn.Module):
    def __init__(self, brain_weights):
        super(FlyBrainAI, self).__init__()
        self.fc = nn.Linear(3, 3, bias=False)
        self.fc.weight = nn.Parameter(brain_weights)
        self.activation = nn.Tanh()

    def forward(self, x):
        return self.activation(x @ self.fc.weight)

# 가상의 시냅스 가중치 설정
synapse_weights = torch.tensor([
    [0.0, 1.5, 0.2],
    [0.5, 0.0, 2.0],
    [1.0, 0.1, 0.0]
], dtype=torch.float32)

ai_brain = FlyBrainAI(synapse_weights)

# --- 2. Pygame(고양이 모니터 세상) 초기화 ---
pygame.init()
WIDTH, HEIGHT = 600, 400
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("초파리 뇌를 가진 반려 고양이 AI")
clock = pygame.time.Clock()

# 고양이(에이전트)의 초기 위치와 속도
cat_x, cat_y = WIDTH // 2, HEIGHT // 2

print("초파리 뇌 반려 고양이 시뮬레이션을 시작합니다!")

while True:
    # 이벤트 처리 (창 닫기 등)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    # --- 3. 외부 환경 자극 생성 (예: 무작위 센서 입력 또는 마우스 위치) ---
    # 여기서는 매 프레임마다 무작위 센서 자극이 초파리 뇌로 들어간다고 가정합니다.
    sensory_input = torch.tensor([[random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)]], dtype=torch.float32)

    # --- 4. 초파리 뇌 모델 구동하여 반응 얻기 ---
    brain_response = ai_brain(sensory_input)
    # 반응 값의 평균을 내어 고양이의 이동 속도나 활기로 변환
    activity_level = brain_response.mean().item()

    # --- 5. 고양이의 행동 결정 및 움직임 업데이트 ---
    # 신경 반응이 활발할수록 고양이가 빠르게 움직이도록 설정
    speed = abs(activity_level) * 5.0
    cat_x += random.choice([-1, 1]) * speed
    cat_y += random.choice([-1, 1]) * speed

    # 화면 밖으로 나가지 않도록 제한
    cat_x = max(20, min(WIDTH - 20, cat_x))
    cat_y = max(20, min(HEIGHT - 20, cat_y))

    # --- 6. 화면 그리기 ---
    screen.fill((30, 30, 30)) # 어두운 배경
    
    # 고양이를 나타내는 원 그리기 (활성도에 따라 색상 변화: 활발하면 노란색, 차분하면 파란색)
    color = (255, 220, 100) if activity_level > 0 else (100, 150, 255)
    pygame.draw.circle(screen, color, (int(cat_x), int(cat_y)), 15)
    
    # 화면 업데이트
    pygame.display.flip()
    clock.tick(30) # 초당 30프레임