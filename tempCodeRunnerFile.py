import tkinter as tk
import random
import math
import os
import torch
import torch.nn as nn

# --- 1. 초파리 뇌 AI 모델 정의 (PyTorch) ---
class FlyBrainAI(nn.Module):
    def __init__(self, brain_weights):
        super(FlyBrainAI, self).__init__()
        self.fc = nn.Linear(3, 3, bias=False)
        self.fc.weight = nn.Parameter(brain_weights)
        self.activation = nn.Tanh()

    def forward(self, x):
        # 초파리 뇌의 신경 회로를 통과해 활성도 계산
        return self.activation(x @ self.fc.weight)

# 실제 시냅스 연결 가중치 행렬 (초파리 뇌의 연결 지도를 흉내 낸 가중치)
synapse_weights = torch.tensor([
    [0.0, 1.8, 0.5],
    [0.7, 0.0, 2.2],
    [1.2, 0.3, 0.0]
], dtype=torch.float32)

ai_brain = FlyBrainAI(synapse_weights)

# --- 2. macOS 투명 창 및 데스크톱 펫 설정 ---
root = tk.Tk()
root.overrideredirect(True)          # 창 틀 없애기
root.wm_attributes("-topmost", True)   # 항상 위에 표시
root.wm_attributes("-transparent", True) # 배경 투명화

SCREEN_WIDTH = root.winfo_screenwidth()
SCREEN_HEIGHT = root.winfo_screenheight()
root.geometry(f"{SCREEN_WIDTH}x{SCREEN_HEIGHT}+0+0")

canvas = tk.Canvas(root, width=SCREEN_WIDTH, height=SCREEN_HEIGHT, highlightthickness=0)
canvas.pack(fill="both", expand=True)
canvas.configure(bg='systemTransparent')

# --- 3. 고양이 및 사료 초기화 ---
cat_x, cat_y = SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2
vx, vy = 0.0, 0.0
wander_timer = 0
target_vx, target_vy = 0.0, 0.0

# 우측 하단 모서리 사료 존
ZONE_X = SCREEN_WIDTH - 250
ZONE_Y = SCREEN_HEIGHT - 250
food_x, food_y = ZONE_X + 125, ZONE_Y + 125
is_dragging_food = False

# 이미지 로드
cat_photo = None
if os.path.exists("cat.png"):
    from PIL import Image, ImageTk
    img = Image.open("cat.png").resize((60, 60))
    cat_photo = ImageTk.PhotoImage(img)

if cat_photo:
    cat_id = canvas.create_image(cat_x, cat_y, image=cat_photo)
else:
    cat_id = canvas.create_oval(cat_x-25, cat_y-25, cat_x+25, cat_y+25, fill="orange")

zone_bg = canvas.create_rectangle(ZONE_X, ZONE_Y, SCREEN_WIDTH-20, SCREEN_HEIGHT-20, fill="#f0e6d2", outline="#d2b48c", width=2)
zone_text = canvas.create_text(ZONE_X + 115, ZONE_Y + 20, text="[ 사료 존 (드래그 가능) ]", font=("Arial", 11, "bold"), fill="#6b4423")
food_id = canvas.create_oval(food_x-10, food_y-10, food_x+10, food_y+10, fill="#ff8c00", outline="#cc7000", width=2)

# --- 4. 드래그 앤 드롭 이벤트 ---
def on_press(event):
    global is_dragging_food
    dist_to_food = math.hypot(event.x - food_x, event.y - food_y)
    if dist_to_food < 20:
        is_dragging_food = True

def on_drag(event):
    global food_x, food_y
    if is_dragging_food:
        food_x, food_y = event.x, event.y
        canvas.coords(food_id, food_x-10, food_y-10, food_x+10, food_y+10)

def on_release(event):
    global is_dragging_food
    is_dragging_food = False

canvas.bind("<Button-1>", on_press)
canvas.bind("<B1-Motion>", on_drag)
canvas.bind("<ButtonRelease-1>", on_release)

# --- 5. 초파리 뇌 AI가 결합된 메인 루프 ---
def update_simulation():
    global cat_x, cat_y, vx, vy, wander_timer, target_vx, target_vy, food_x, food_y

    dist_to_food = math.hypot(food_x - cat_x, food_y - cat_y)

    if dist_to_food < 400 and not is_dragging_food:
        dx = food_x - cat_x
        dy = food_y - cat_y
        dist = math.hypot(dx, dy)
        
        if dist > 5:
            # 사료와의 거리 및 방향 정보를 초파리 뇌 AI 모델의 자극(Sensory Input)으로 입력
            sensory_input = torch.tensor([[dx/dist, dy/dist, dist/300.0]], dtype=torch.float32)
            
            # 초파리 뇌 회로 구동하여 신경 반응(Activity) 얻기
            brain_response = ai_brain(sensory_input)
            activity = brain_response.mean().item()

            # 초파리 뇌의 신경 반응 수치에 따라 이동 속도와 활기도 조절
            speed = 2.0 * (1.0 + abs(activity))
            target_vx = (dx / dist) * speed
            target_vy = (dy / dist) * speed
        
        # 사료에 도착하면 사료를 먹고 원래 자리로 리셋
        if dist < 20:
            food_x, food_y = ZONE_X + 125, ZONE_Y + 125
            canvas.coords(food_id, food_x-10, food_y-10, food_x+10, food_y+10)
    else:
        # 자율 산책 모드
        wander_timer -= 1
        if wander_timer <= 0:
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(0.6, 1.5)
            target_vx = math.cos(angle) * speed
            target_vy = math.sin(angle) * speed
            wander_timer = random.randint(120, 300)

        # 평소 뇌 자극 시뮬레이션
        sensory_input = torch.tensor([[random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1), 0.5]], dtype=torch.float32)
        _ = ai_brain(sensory_input)

    # 관성(부드러운 움직임) 적용
    vx += (target_vx - vx) * 0.1
    vy += (target_vy - vy) * 0.1
    cat_x += vx
    cat_y += vy

    # 화면 경계 처리
    if cat_x < 50 or cat_x > SCREEN_WIDTH - 50:
        target_vx *= -1
        vx *= -1
    if cat_y < 50 or cat_y > SCREEN_HEIGHT - 50:
        target_vy *= -1
        vy *= -1

    cat_x = max(50, min(SCREEN_WIDTH - 50, cat_x))
    cat_y = max(50, min(SCREEN_HEIGHT - 50, cat_y))

    # 고양이 화면 위치 갱신
    if cat_photo:
        canvas.coords(cat_id, cat_x, cat_y)
    else:
        canvas.coords(cat_id, cat_x-25, cat_y-25, cat_x+25, cat_y+25)

    root.after(30, update_simulation)

root.after(30, update_simulation)
root.mainloop()