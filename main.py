import io
import os
import math
import uuid
import sqlite3
import json
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v3_large

app = FastAPI(title="Bird Recognition API")

# 创建用于保存用户上传图片的文件夹，并配置静态目录映射
os.makedirs("uploads", exist_ok=True)
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# 允许前端跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 初始化 SQLite 数据库
def init_db():
    conn = sqlite3.connect('bird_history.db')
    c = conn.cursor()
    c.execute('''
              CREATE TABLE IF NOT EXISTS history
              (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  bird_name TEXT,
                  similarity REAL,
                  location TEXT,
                  image_path TEXT,
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
              )
              ''')
    conn.commit()
    conn.close()


init_db()


# 加载鸟类详细信息
def load_bird_details():
    details_path = os.path.join(os.path.dirname(__file__), 'bird_details.json')
    try:
        with open(details_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

BIRD_DETAILS = load_bird_details()


# 读取 class_names.txt 中的真实鸟类名称
def load_class_names():
    # 读取当前目录下的 class_names.txt
    file_path = os.path.join(os.path.dirname(__file__), 'class_names.txt')
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            names = []
            for line in f.readlines():
                line = line.strip()
                if not line: continue

                if ',' in line:
                    en_name, zh_name = line.split(',', 1)
                    names.append(f"{zh_name.strip()} ({en_name.strip()})")
                else:
                    names.append(line)
            return names
    except FileNotFoundError:
        return [f"Bird_Class_{i}" for i in range(325)]


CLASS_NAMES = load_class_names()
NUM_CLASSES = len(CLASS_NAMES)  # 自动识别出共有 275 个类别


# 加载 MobileNetV3 模型
def get_model():
    model = mobilenet_v3_large(weights=None)
    # 修改最后一层以适应你的分类数
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, NUM_CLASSES)

    # 加载你刚刚训练好的模型权重
    import os
    model_weights_path = os.path.join(os.path.dirname(__file__), 'datasets', 'best_mobilenetv3.pth')
    model.load_state_dict(torch.load(model_weights_path, map_location=torch.device('cpu')))

    model.eval()
    return model


model = get_model()

# 定义图像预处理流程
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


@app.get("/")
async def serve_index():
    index_path = os.path.join(os.path.dirname(__file__), 'index.html')
    return FileResponse(index_path)


@app.post("/predict")
async def predict_bird(
    file: UploadFile = File(...), 
    location: str = Form("未知地点")
):
    try:
        # 读取图像
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # 保存图片到本地 uploads 文件夹
        filename = f"{uuid.uuid4().hex}.jpg"
        filepath = os.path.join("uploads", filename)
        with open(filepath, "wb") as f:
            f.write(image_bytes)

        # 预处理
        input_tensor = transform(image).unsqueeze(0)

        # 推理
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            similarity, class_idx = torch.max(probabilities, dim=0)

        bird_name = CLASS_NAMES[class_idx.item()]
        similarity_score = round(similarity.item() * 100, 2)
        image_url = f"/uploads/{filename}"

        # 将记录保存到数据库中
        conn = sqlite3.connect('bird_history.db')
        c = conn.cursor()
        c.execute("INSERT INTO history (bird_name, similarity, location, image_path) VALUES (?, ?, ?, ?)",
                  (bird_name, similarity_score, location, image_url))
        conn.commit()
        conn.close()

        return {
            "bird_name": bird_name,
            "similarity": similarity_score,
            "location": location,
            "image_path": image_url
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})


@app.get("/bird_details/{bird_name}")
async def get_bird_details(bird_name: str):
    details = BIRD_DETAILS.get(bird_name)
    if details:
        details["full_name"] = bird_name # 返回匹配到的完整名称
        return details

    query = bird_name.lower()
    for full_name, data in BIRD_DETAILS.items():
        if query in full_name.lower():
            matched_data = data.copy()
            matched_data["full_name"] = full_name
            return matched_data

    raise HTTPException(status_code=404, detail="未找到该鸟类的详细信息")


# 获取同类鸟的发现记录
@app.get("/history/same_bird")
async def get_same_bird_history(bird_name: str):
    conn = sqlite3.connect('bird_history.db')
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM history WHERE bird_name = ? ORDER BY timestamp DESC LIMIT 6", (bird_name,))
    rows = c.fetchall()
    conn.close()
    return {"history": [dict(row) for row in rows]}


@app.get("/history")
async def get_history(query: str = "", page: int = 1, limit: int = 8):
    # 获取记录，支持分页和按鸟类名称模糊搜索
    conn = sqlite3.connect('bird_history.db')
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    
    offset = (page - 1) * limit
    
    if query:
        c.execute("SELECT COUNT(*) FROM history WHERE bird_name LIKE ?", (f"%{query}%",))
        total_count = c.fetchone()[0]
        c.execute("SELECT * FROM history WHERE bird_name LIKE ? ORDER BY timestamp DESC LIMIT ? OFFSET ?", (f"%{query}%", limit, offset))
    else:
        c.execute("SELECT COUNT(*) FROM history")
        total_count = c.fetchone()[0]
        c.execute("SELECT * FROM history ORDER BY timestamp DESC LIMIT ? OFFSET ?", (limit, offset))
        
    rows = c.fetchall()
    conn.close()

    history = []
    for row in rows:
        row_dict = dict(row)
        history.append({
            "bird_name": row_dict.get("bird_name"),
            "similarity": row_dict.get("similarity"),
            "location": row_dict.get("location", "未知地点"),
            "image_path": row_dict.get("image_path", ""),
            "timestamp": row_dict.get("timestamp")
        })
        
    total_pages = math.ceil(total_count / limit) if total_count > 0 else 1
    
    return {
        "history": history,
        "total_pages": total_pages,
        "current_page": page
    }