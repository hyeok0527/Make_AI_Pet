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

# 시냅스 가중치 설정 (초파리 뇌 회로 모사)
synapse_weights = torch.tensor([
    [0.0, 1.8, 0.5],
    [0.7, 0.0, 2.2],
    [1.2, 0.3, 0.0]
], dtype=torch.float32)

ai_brain = FlyBrainAI(synapse_weights)

# --- 2. Pygame 초기화 ---
pygame.init()
WIDTH, HEIGHT = 900, 700
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("반려 고양이 키우기")
clock = pygame.time.Clock()

# 고양이 상태 변수
cat_x, cat_y = WIDTH / 2, HEIGHT / 2
vx, vy = 0.0, 0.0
cat_hunger = 50.0  # 배고픔 수치 (0이면 배부름, 100이면 매우 배고픔)
wander_timer = 0
target_vx, target_vy = 0.0, 0.0

# 사료 시스템 (사료 리스트: 각 사료의 [x, y] 좌표)
foods = []
dragging_food = False

# 고양이 이미지 로드
cat_image = None
if os.path.exists("cat.png"):
    img = pygame.image.load("cat.png")
    cat_image = pygame.transform.scale(img, (60, 60))

print("초파리 뇌 반려 고양이 시뮬레이션 시작!")
print("- 마우스 좌클릭 드래그: 사료 주기")
print("- 마우스 우클릭 유지: 장난감 흔들기")

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        
        # 마우스로 사료 던지기 (좌클릭 시 사료 생성)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1: # 좌클릭
                mouse_pos = pygame.mouse.get_pos()
                foods.append(list(mouse_pos))

    mouse_x, mouse_y = pygame.mouse.get_pos()
    right_click = pygame.mouse.get_pressed()[2] # 우클릭 여부 (장난감)

    # --- 3. 우선순위 목표 설정 (사료 > 장난감 > 자율 산책) ---
    target_x, target_y = None, None
    stimulus_type = "wander"

    # 1순위: 가장 가까운 사료 찾기
    nearest_food = None
    min_food_dist = float('inf')
    for f in foods:
        dist = math.hypot(f[0] - cat_x, f[1] - cat_y)
        if dist < min_food_dist:
            min_food_dist = dist
            nearest_food = f

    if nearest_food and min_food_dist < 400:
        target_x, target_y = nearest_food[0], nearest_food[1]
        stimulus_type = "food"
    elif right_click:
        # 2순위: 우클릭 중이면 마우스 위치가 장난감 역할
        target_x, target_y = mouse_x, mouse_y
        stimulus_type = "toy"
    else:
        # 3순위: 평소에는 자율 산책
        stimulus_type = "wander"

    # --- 4. 초파리 뇌 신경 자극 입력 ---
    if stimulus_type in ["food", "toy"] and target_x is not None:
        dx = target_x - cat_x
        dy = target_y - cat_y
        dist = math.hypot(dx, dy)
        if dist > 0:
            # 뇌에 자극 전달 (거리와 방향)
            sensory_input = torch.tensor([[dx/dist, dy/dist, dist/300.0]], dtype=torch.float32)
            brain_response = ai_brain(sensory_input)
            activity = brain_response.mean().item()

            # 목표를 향해 다가가는 힘 계산
            speed = 3.0 * (1.0 + abs(activity))
            target_vx = (dx / dist) * speed
            target_vy = (dy / dist) * speed

            # 사료에 아주 가까이 가면 사료 냠냠 먹고 사라짐
            if stimulus_type == "food" and dist < 25:
                foods.remove(nearest_food)
                cat_hunger = max(0.0, cat_hunger - 30.0) # 배고픔 감소
        else:
            target_vx, target_vy = 0, 0
    else:
        # 자율 산책 모드
        wander_timer -= 1
        if wander_timer <= 0:
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(0.5, 1.5)
            target_vx = math.cos(angle) * speed
            target_vy = math.sin(angle) * speed
            wander_timer = random.randint(90, 200)

        sensory_input = torch.tensor([[random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1), 0.5]], dtype=torch.float32)
        _ = ai_brain(sensory_input)

    # 시간에 따라 천천히 배고픔 증가
    cat_hunger = min(100.0, cat_hunger + 0.02)

    # --- 5. 부드러운 움직임 (관성 적용) ---
    vx += (target_vx - vx) * 0.1
    vy += (target_vy - vy) * 0.1
    cat_x += vx
    cat_y += vy

    # 화면 경계 제한
    cat_x = max(40, min(WIDTH - 40, cat_x))
    cat_y = max(40, min(HEIGHT - 40, cat_y))

    # --- 6. 화면 그리기 (렌더링) ---
    screen.fill((245, 243, 238)) # 아늑한 방 배경 색상

    # 사료 그리기 (노란색 알갱이)
    for f in foods:
        pygame.draw.circle(screen, (220, 140, 50), (int(f[0]), int(f[1])), 8)

    # 우클릭 중이면 장난감(깃털/빨간 점) 표시
    if right_click:
        pygame.draw.circle(screen, (230, 60, 60), (mouse_x, mouse_y), 10)
        # 깃털 장식 느낌의 선
        pygame.draw.line(screen, (150, 150, 150), (mouse_x, mouse_y), (mouse_x + 15, mouse_y - 15), 3)

    # 고양이 그리기
    if cat_image:
        rect = cat_image.get_rect(center=(int(cat_x), int(cat_y)))
        screen.blit(cat_image, rect.topleft)
    else:
        pygame.draw.circle(screen, (255, 165, 0), (int(cat_x), int(cat_y)), 25)

    # 상단에 상태 정보 텍스트 표시 (배고픔 등)
    font = pygame.font.SysFont(None, 24)
    hunger_text = font.render(f"Cat Hunger (배고픔): {int(cat_hunger)}%", True, (80, 80, 80))
    guide_text = font.render("좌클릭: 사료 주기 | 우클릭 유지: 장난감 흔들기", True, (120, 120, 120))
    screen.blit(hunger_text, (20, 20))
    screen.blit(guide_text, (20, 50))

    pygame.display.flip()
    clock.tick(60)