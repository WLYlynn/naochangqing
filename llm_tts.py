# llm_tts.py
import requests
import json
import base64
import time
import uuid
import threading
import websocket
import pyaudio
from websocket import create_connection, ABNF

# 这里也要定义 APP_KEY 等配置，或者从主程序传入
APP_KEY = "sk-xuanji-2026525296-T1VZV2pHS3hJVXpZcXZOZg=="
TTS_ENGINE_ID = "short_audio_synthesis_jovi"
TTS_VCN = "vivoHelper"
TTS_SPEED = 50
TTS_VOLUME = 80

def call_llm(user_message, history=None):
    """调用 vivo 大模型进行多轮对话，返回回复文本"""
    if history is None:
        history = []

    messages = [
        {
            "role": "system",
            "content": (
                "你是一位老年认知评估助手，负责与老人进行自然闲聊，在对话中无感地评估其认知状态。\n"
                "请用亲切、耐心的语气回应，适当追问记忆、逻辑、情绪相关的问题。\n"
                "不要直接提'认知评估'，保持日常闲聊氛围。\n"
                "每次回答控制在 1-2 句话，口语化，适合语音合成。"
            )
        }
    ]
    messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    url = "https://api-ai.vivo.com.cn/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {APP_KEY}"
    }
    data = {
        "model": "Doubao-Seed-2.0-lite",
        "messages": messages,
        "stream": False,
        "max_tokens": 1024,
        "temperature": 0.7,
        "reasoning_effort": "medium"
    }
    try:
        response = requests.post(url, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        result = response.json()
        reply = result["choices"][0]["message"]["content"]
        return reply
    except Exception as e:
        print(f"[LLM] 请求失败: {e}")
        return "我这边有点卡顿，您能再说一遍吗？"

def tts_synthesize(text, engine_id=TTS_ENGINE_ID, vcn=TTS_VCN, speed=TTS_SPEED, volume=TTS_VOLUME, interrupt_event=None):
    """
        调用 vivo TTS WebSocket 接口，合成语音并实时播放
        （参考官方示例 tts_examples.py）
        """
    import websocket as ws_lib  # 避免冲突
    from websocket import create_connection, ABNF

    # 构建 URL 参数
    system_time = str(int(time.time()))
    user_id = uuid.uuid4().hex[:32]
    request_id = str(uuid.uuid4())

    params = {
        "engineid": engine_id,
        "system_time": system_time,
        "user_id": user_id,
        "model": "PC",
        "product": "unknown",
        "package": "com.vivo.tts.demo",
        "client_version": "1.0.0",
        "system_version": "unknown",
        "sdk_version": "3.0",
        "android_version": "unknown",
        "requestId": request_id
    }
    param_str = "&".join([f"{k}={v}" for k, v in params.items()])
    ws_url = f"wss://api-ai.vivo.com.cn/tts?{param_str}"

    # 请求头（增加 vaid 字段，参考官方示例）
    headers = {
        "Authorization": f"Bearer {APP_KEY}",
        "X-AI-GATEWAY-SIGNATURE": "developers-aigc",
        "vaid": "123456789"   # 官方示例中出现，虽未在文档提及但建议加上
    }

    try:
        # 1. 建立 WebSocket 连接（使用 create_connection）
        ws = create_connection(ws_url, header=headers)
        print("[TTS] WebSocket 连接建立成功")

        # 2. 接收握手确认（官方示例做了这一步）
        code, data = ws.recv_data(True)
        try:
            handshake = json.loads(data)
            if handshake.get("error_code") != 0:
                print(f"[TTS] 握手失败: {handshake}")
                return None
            else:
                print("[TTS] 握手确认:", handshake.get("error_msg"))
        except:
            print("[TTS] 握手数据非 JSON，原始:", data[:100])

        # 3. 发送合成请求（包含 sfl 字段）
        text_b64 = base64.b64encode(text.encode('utf-8')).decode('utf-8')
        req = {
            "aue": 0,
            "auf": "audio/L16;rate=24000",
            "vcn": vcn,
            "speed": speed,
            "volume": volume,
            "text": text_b64,
            "encoding": "utf8",
            "reqId": int(time.time() * 1000),
            "sfl": 1   # 官方示例中有此字段
        }
        ws.send(json.dumps(req))
        print(f"[TTS] 已发送合成请求: {text[:30]}...")

        # 4. 循环接收音频数据
        audio_frames = []
        while True:
            code, data = ws.recv_data(True)
            if code == ABNF.OPCODE_CLOSE:
                print("[TTS] 服务端关闭连接")
                break
            elif code == ABNF.OPCODE_TEXT:
                try:
                    resp = json.loads(data)
                    if resp.get("error_code") != 0:
                        print(f"[TTS] 合成错误: {resp}")
                        break
                    # 提取音频
                    audio_b64 = resp.get("data", {}).get("audio", "")
                    if audio_b64:
                        audio_bytes = base64.b64decode(audio_b64)
                        audio_frames.append(audio_bytes)
                    # 检查结束
                    if resp.get("data", {}).get("status") == 2:
                        print("[TTS] 合成完成")
                        break
                except Exception as e:
                    print(f"[TTS] 解析响应异常: {e}")
                    break
            else:
                print(f"[TTS] 未知帧类型: {code}")
                break

        ws.close()

        # 5. 合并音频并播放
        if audio_frames:
            full_audio = b''.join(audio_frames)
            if interrupt_event:
                interrupt_event.clear()   # 清除旧打断信号
            interrupted = play_pcm_with_interrupt(full_audio, sample_rate=24000, interrupt_event=interrupt_event)

    except Exception as e:
        print(f"[TTS] 连接或合成异常: {repr(e)}")
        return None

def play_pcm_with_interrupt(pcm_data, sample_rate=24000, interrupt_event=None):
    """
    播放 PCM 音频，并定期检查 interrupt_event 是否被设置，如果是则停止播放。
    """
    import pyaudio

    CHUNK = 4096
    p = pyaudio.PyAudio()
    play_stream = p.open(
        format=pyaudio.paInt16, channels=1, rate=sample_rate, output=True
    )

    data_bytes = pcm_data
    total_bytes = len(data_bytes)
    pos = 0
    interrupted = False

    try:
        while pos < total_bytes:
            # 检查打断信号
            if interrupt_event and interrupt_event.is_set():
                print("[TTS] 检测到打断信号，停止播放")
                interrupted = True
                break

            chunk = data_bytes[pos : pos + CHUNK * 2]
            if chunk:
                play_stream.write(chunk)
            pos += CHUNK * 2

    finally:
        play_stream.stop_stream()
        play_stream.close()
        p.terminate()

    return interrupted