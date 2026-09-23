import torch
import torch.nn as nn
import pygame
import math
import sys
import os
import random

# --- 1. 초파리 뇌 AI 모델 정의 ---
class FlyBrainAI(nn.Module):
    def __init__(self, brain_weights):
        super(FlyBrainAI, self).__init__()
        self.fc = nn.Linear(3, 3, bias=False)
        self.fc.weight = nn.Parameter(brain_weights)
        self.activation = nn.Tanh()

    def forward(self, x):
        return self.activation(x @ self.fc.weight)

synapse_weights = torch.tensor([
    [0.0, 1.5, 0.2],
    [0.5, 0.0, 2.0],
    [1.0, 0.1, 0.0]
], dtype=torch.float32)

ai_brain = FlyBrainAI(synapse_weights)

# --- 2. Pygame 초기화 ---
pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("자율적으로 움직이는 초파리 뇌 고양이 AI")
clock = pygame.time.Clock()

# 고양이 위치 및 속도 벡터 (관성 부여)
cat_x, cat_y = WIDTH / 2, HEIGHT / 2
vx, vy = 0.0, 0.0

# 자율 산책을 위한 타이머 및 목표 방향 변수
wander_timer = 0
target_vx, target_vy = 0.0, 0.0

# 고양이 이미지 로드
cat_image = None
if os.path.exists("cat.png"):
    img = pygame.image.load("cat.png")
    cat_image = pygame.transform.scale(img, (60, 60))

print("자율 행동하는 초파리 고양이 AI 시뮬레이션 시작!")

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    mouse_x, mouse_y = pygame.mouse.get_pos()
    dx = mouse_x - cat_x
    dy = mouse_y - cat_y
    distance = math.hypot(dx, dy)

    # --- 3. 초파리 뇌 자극 및 상태 결정 ---
    if distance < 200 and distance > 0:
        # [경계 모드] 마우스가 가까우면 위협 자극 입력
        sensory_input = torch.tensor([[-dx/distance, -dy/distance, distance/200.0]], dtype=torch.float32)
        brain_response = ai_brain(sensory_input)
        activity = brain_response.mean().item()

        # 마우스 반대 방향으로 도망치는 힘 계산
        escape_force = (200 - distance) / 50.0 * (1.0 + abs(activity))
        target_vx = -(dx / distance) * escape_force * 3
        target_vy = -(dy / distance) * escape_force * 3
    else:
        # [자율 산책 모드] 마우스가 멀리 있으면 고양이가 혼자서 돌아다님
        wander_timer -= 1
        if wander_timer <= 0:
            # 일정 시간마다 무작위 방향과 속도 설정 (심심하면 방향 전환)
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(0.5, 2.0)
            target_vx = math.cos(angle) * speed
            target_vy = math.sin(angle) * speed
            wander_timer = random.randint(60, 180) # 1~3초마다 행동 변경

        # 평소 초파리 뇌의 잔잔한 신경 반응
        sensory_input = torch.tensor([[random.uniform(-0.2, 0.2), random.uniform(-0.2, 0.2), 0.5]], dtype=torch.float32)
        brain_response = ai_brain(sensory_input)

    # --- 4. 부드러운 움직임 (관성/가속도 적용) ---
    # 목표 속도(target_vx)를 향해 현재 속도(vx)가 부드럽게 쫓아감 (Lerp 방식)
    vx += (target_vx - vx) * 0.1
    vy += (target_vy - vy) * 0.1

    cat_x += vx
    cat_y += vy

    # 화면 경계에 부딪히면 안쪽으로 방향 틀어주기
    if cat_x < 40 or cat_x > WIDTH - 40:
        target_vx *= -1
        vx *= -1
    if cat_y < 40 or cat_y > HEIGHT - 40:
        target_vy *= -1
        vy *= -1

    cat_x = max(40, min(WIDTH - 40, cat_x))
    cat_y = max(40, min(HEIGHT - 40, cat_y))

    # --- 5. 화면 렌더링 ---
    screen.fill((240, 240, 250))
    
    # 마우스 커서 표시
    pygame.draw.circle(screen, (255, 100, 100), (mouse_x, mouse_y), 6)

    # 고양이 출력
    if cat_image:
        rect = cat_image.get_rect(center=(int(cat_x), int(cat_y)))
        screen.blit(cat_image, rect.topleft)
    else:
        pygame.draw.circle(screen, (255, 150, 50), (int(cat_x), int(cat_y)), 25)

    pygame.display.flip()
    clock.tick(60)