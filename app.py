# -*- coding: utf-8 -*-
"""
FastAPI 服务：连接前端与后端
前端页面在 static/ 文件夹中，通过 API 调用后端功能
"""

import os
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

# 导入你的对话管理器
from dialogue_manager import DialogueManager

from datetime import datetime

# ==================== 创建 FastAPI 应用 ====================
app = FastAPI(title="脑长青后端服务")

# 允许跨域请求（让前端能调用后端）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # 允许所有来源
    allow_credentials=True,
    allow_methods=["*"],           # 允许所有 HTTP 方法
    allow_headers=["*"],           # 允许所有请求头
)

# 1. 挂载静态文件目录（前端页面、图片等）
app.mount("/static", StaticFiles(directory="static"), name="static")

# 2. 创建全局对话管理器
dm = DialogueManager()

# ==================== 房间管理 ====================

import json
import random
import string

ROOMS_DIR = "rooms"
os.makedirs(ROOMS_DIR, exist_ok=True)

def generate_room_code():
    """生成6位数字房间码"""
    return ''.join(random.choices(string.digits, k=6))

def save_room(room_code, data):
    """保存房间数据到文件"""
    file_path = os.path.join(ROOMS_DIR, f"{room_code}.json")
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_room(room_code):
    """读取房间数据"""
    file_path = os.path.join(ROOMS_DIR, f"{room_code}.json")
    if not os.path.exists(file_path):
        return None
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# ==================== 房间管理 API ====================

@app.post("/api/create_room")
async def create_room():
    """家属端创建房间，返回6位房间码"""
    room_code = generate_room_code()
    while load_room(room_code) is not None:
        room_code = generate_room_code()

    room_data = {
        "created_at": datetime.now().isoformat(),
        "paired": False,
        "elderly_joined": False,
        "reports": []
    }
    save_room(room_code, room_data)
    return {"success": True, "room_code": room_code}


@app.post("/api/join_room")
async def join_room(request: dict):
    """老人端加入房间，验证房间码是否存在"""
    room_code = request.get("room_code", "").strip()
    if not room_code or len(room_code) != 6:
        raise HTTPException(status_code=400, detail="房间码格式错误")

    room_data = load_room(room_code)
    if room_data is None:
        raise HTTPException(status_code=404, detail="房间码不存在，请检查输入")

    room_data["elderly_joined"] = True
    save_room(room_code, room_data)
    return {"success": True, "message": "加入房间成功"}


@app.post("/api/end_dialogue_with_room")
async def end_dialogue_with_room(request: dict):
    """
    老人端结束对话，生成报告并关联到房间码
    """
    room_code = request.get("room_code", "").strip()
    if not room_code or len(room_code) != 6:
        raise HTTPException(status_code=400, detail="房间码格式错误")

    room_data = load_room(room_code)
    if room_data is None:
        raise HTTPException(status_code=404, detail="房间码不存在")

    try:
        dm.end_dialogue()
        reports = []
        if os.path.exists("output"):
            reports = [f for f in os.listdir("output") if f.endswith(".txt")]
        if reports:
            latest_report = reports[-1]
            room_data["reports"].append(latest_report)
            save_room(room_code, room_data)
        return {"success": True, "message": "对话结束，报告已关联", "reports": reports}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/room_reports/{room_code}")
async def get_room_reports(room_code: str):
    """家属端获取指定房间的所有报告"""
    room_data = load_room(room_code)
    if room_data is None:
        raise HTTPException(status_code=404, detail="房间码不存在")

    reports = []
    for filename in room_data.get("reports", []):
        file_path = os.path.join("output", filename)
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            reports.append({
                "filename": filename,
                "content": content
            })
    return {"success": True, "reports": reports}

# ==================== API 接口 ====================
@app.get("/api/reports")
async def get_reports():
    """返回 output/ 目录下所有报告文件列表"""
    reports = []
    if os.path.exists("output"):
        for f in os.listdir("output"):
            if f.endswith(".txt"):
                file_path = os.path.join("output", f)
                # 获取文件修改时间
                mtime = os.path.getmtime(file_path)
                reports.append({
                    "filename": f,
                    "timestamp": datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
                })
    # 按时间倒序排列（最新的在前）
    reports.sort(key=lambda x: x["timestamp"], reverse=True)
    return {"reports": reports}

@app.get("/api/report/{filename}")
async def get_report(filename: str):
    """返回指定报告文件的内容"""
    file_path = os.path.join("output", filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="报告不存在")
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    return {"filename": filename, "content": content}


@app.get("/api/status")
async def get_status():
    return {
        "is_dialogue_active": dm.is_dialogue_active,
        "is_recording": dm.is_recording,
        "current_user_text": dm.current_user_text,
        "current_reply": dm.current_reply,
    }


@app.post("/api/start_dialogue")
async def start_dialogue():
    try:
        dm.start_dialogue()
        return {"success": True, "message": "对话已开始"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/start_recording")
async def start_recording():
    try:
        dm.start_recording()
        return {"success": True, "message": "录音已开始"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/stop_recording")
async def stop_recording():
    try:
        user_text, reply = dm.stop_recording()
        return {"success": True, "user_text": user_text, "reply": reply}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/end_dialogue")
async def end_dialogue():
    try:
        dm.end_dialogue()
        reports = []
        if os.path.exists("output"):
            reports = [f for f in os.listdir("output") if f.endswith(".txt")]
        return {"success": True, "message": "对话结束，报告已生成", "reports": reports}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 页面路由（返回完整的 HTML 文件） ====================

@app.get("/")
async def index():
    return FileResponse("static/index.html")

@app.get("/family")
async def family_page():
    return FileResponse("static/family.html")

@app.get("/elderly")
async def elderly_page():
    return FileResponse("static/elderly.html")


# ==================== 启动服务器 ====================
if __name__ == "__main__":
    print("=" * 60)
    print("🚀 启动脑长青后端服务（分离版）")
    print("📡 访问地址: http://localhost:8000")
    print("📁 前端页面在 static/ 文件夹中")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8000)