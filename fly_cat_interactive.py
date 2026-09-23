import torch
import torch.nn as nn
import pygame
import math
import sys
import os

# --- 1. 초파리 뇌 AI 모델 정의 ---
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

# --- 2. Pygame 초기화 및 화면 설정 ---
pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("초파리 뇌를 가진 반려 고양이 AI (마우스 회피 모드)")
clock = pygame.time.Clock()

# 고양이(에이전트) 초기 위치
cat_x, cat_y = WIDTH // 2, HEIGHT // 2

# --- 3. 고양이 이미지 불러오기 ---
# 같은 폴더에 'cat.png' 파일이 있어야 합니다. 없으면 대체 원을 사용합니다.
cat_image = None
if os.path.exists("cat.png"):
    img = pygame.image.load("cat.png")
    cat_image = pygame.transform.scale(img, (60, 60)) # 크기 조절 (60x60)
else:
    print("경고: 'cat.png' 파일을 찾지 못해 기본 도형으로 대체합니다. 폴더에 고양이 이미지를 넣어주세요!")

print("초파리 뇌 고양이 시뮬레이션 시작!")

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    # --- 4. 센서 입력: 고양이와 마우스 간의 거리 및 방향 계산 ---
    mouse_x, mouse_y = pygame.mouse.get_pos()
    
    # 고양이와 마우스 사이의 거리 계산
    dx = mouse_x - cat_x
    dy = mouse_y - cat_y
    distance = math.hypot(dx, dy)
    
    # 마우스가 너무 가까이 오면(예: 200픽셀 이내) 초파리 뇌에 위협(자극) 입력
    if distance < 200 and distance > 0:
        # 마우스가 가까울수록 강한 자극 입력 (정규화된 방향 벡터 활용)
        sensory_input = torch.tensor([[-dx/distance, -dy/distance, distance/200.0]], dtype=torch.float32)
    else:
        sensory_input = torch.tensor([[0.1, 0.1, 0.1]], dtype=torch.float32)

    # --- 5. 초파리 뇌 모델 구동 ---
    brain_response = ai_brain(sensory_input)
    activity_level = brain_response.mean().item()

    # --- 6. 행동 결정: 마우스 피해서 도망치기 + 초파리 뇌 반응 결합 ---
    if distance < 200 and distance > 0:
        # 마우스 반대 방향으로 도망치는 벡터 계산
        escape_speed = (200 - distance) / 20.0 * (1.0 + abs(activity_level))
        cat_x -= (dx / distance) * escape_speed
        cat_y -= (dy / distance) * escape_speed
    else:
        # 평소에는 천천히 무작위로 배회
        pass

    # 화면 밖으로 나가지 않도록 제한
    cat_x = max(30, min(WIDTH - 30, cat_x))
    cat_y = max(30, min(HEIGHT - 30, cat_y))

    # --- 7. 화면 그리기 ---
    screen.fill((240, 240, 250)) # 부드러운 배경 색상
    
    # 마우스 위치 표시 (빨간 작은 점)
    pygame.draw.circle(screen, (255, 100, 100), (mouse_x, mouse_y), 6)

    # 고양이 렌더링 (이미지가 있으면 이미지, 없으면 사각형으로 표시)
    if cat_image:
        # 이미지의 중심이 좌표에 오도록 위치 조정
        rect = cat_image.get_rect(center=(int(cat_x), int(cat_y)))
        screen.blit(cat_image, rect.topleft)
    else:
        # 대체 도형 (초파리 뇌 활성도에 따라 색상 변화)
        color = (255, 150, 50) if distance < 200 else (100, 200, 100)
        pygame.draw.circle(screen, color, (int(cat_x), int(cat_y)), 25)

    # 안내 텍스트나 상태 표시 추가 가능
    pygame.display.flip()
    clock.tick(60) # 초당 60프레임으로 부드럽게 움직임