import struct
import time
import uuid
import os
import threading
import websocket
import numpy as np
import soundfile as sf
import json
import requests
import base64
import pyaudio          # 如果已有，无需重复
import wave
import io
import queue
from datetime import datetime
# 在文件顶部导入
from extract_features import extract_audio_features


# ================== 配置 ==================
APP_KEY = "sk-xuanji-2026525296-T1VZV2pHS3hJVXpZcXZOZg=="
# ================== TTS 配置 ==================
TTS_ENGINE_ID = "short_audio_synthesis_jovi"  # 短音频合成（对话用）
TTS_VCN = "vivoHelper"                         # 音色：奕雯（温柔女声）
TTS_SPEED = 50                                 # 语速 0-100
TTS_VOLUME = 80                                # 音量 1-100
# =============================================
ENGINE_ID = "longasrlisten"   # ✅ 长语音能力ID
USER_ID = uuid.uuid4().hex[:32]
# =========================================

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
    play_stream = p.open(format=pyaudio.paInt16,
                         channels=1,
                         rate=sample_rate,
                         output=True)

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

            chunk = data_bytes[pos:pos+CHUNK*2]
            if chunk:
                play_stream.write(chunk)
            pos += CHUNK*2

    finally:
        play_stream.stop_stream()
        play_stream.close()
        p.terminate()

    return interrupted

class VivoLongASRClient:
    def __init__(self, app_key, engine_id="longasrlisten", user_id=None):
        self.app_key = app_key
        self.engine_id = engine_id
        self.user_id = user_id or USER_ID
        self.ws = None
        self.result_text = ""
        self.is_finished = False
        self.request_id = str(uuid.uuid4()).replace("-", "")[:32]
        self._stop_send = False
        self.audio_data = bytearray()          # 存储原始 PCM 数据
        self.audio_saved_path = None           # 保存的 WAV 文件路径
        self.features = None                   # 提取的特征字典
        self.audio_data_lock = threading.Lock()  # 保证数据读写安全
        self.current_features = None             # 实时更新的特征

        self.interrupt_event = threading.Event()   # 打断事件
        self.is_playing_tts = False                # 是否正在播放 TTS


    def _build_handshake_url(self):
        system_time = str(int(time.time() * 1000))
        params = {
            "client_version": "1.0.0",
            "package": "com.vivo.asr.demo",
            "sdk_version": "3.0",
            "user_id": self.user_id,
            "android_version": "unknown",
            "system_time": system_time,
            "net_type": "1",
            "engineid": self.engine_id,
            "requestId": self.request_id
        }
        param_str = "&".join([f"{k}={v}" for k, v in params.items()])
        return f"ws://api-ai.vivo.com.cn/asr/v2?{param_str}"

    def on_message(self, ws, message):
        try:
            data = json.loads(message)
            action = data.get("action")

            if action == "started":
                print("[ASR] ✅ 握手成功，连接已建立")
                return

            if action == "result":
                code = data.get("code", -1)
                result_data = data.get("data", {})

                if code == 8:   # 中间结果（var）
                    var_text = result_data.get("var", "")
                    if var_text:
                        print(f"[ASR] 📝 中间: {var_text}")

                elif code == 0: # 完整句（onebest）
                    onebest = result_data.get("onebest", "")
                    if onebest:
                        self.result_text += onebest
                        print(f"[ASR] ✅ 完整句: {onebest}")


                elif code == 9:
                    onebest = result_data.get("onebest", "")
                    if onebest:
                        self.result_text += onebest
                    print(f"[ASR] 🏁 识别结束（最后一句）")
                    self.is_finished = True
                    self._stop_send = True
                    self.ws.close()   # 主动关闭连接，让 run_forever 退出
                return

            if action == "error":
                print(f"[ASR] ❌ 错误: {data}")
                self.is_finished = True
                return

            print(f"[ASR] 其他消息: {data}")

        except json.JSONDecodeError:
            print(f"[ASR] 非JSON消息: {message}")

    def on_error(self, ws, error):
        print(f"[ASR] ❌ 连接错误: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        print("[ASR] 🔌 连接已关闭")

    def on_open(self, ws):
        print("[ASR] 🔗 连接已打开，发送 started 指令...")

        start_msg = {
            "type": "started",
            "request_id": self.request_id,
            "asr_info": {
                "front_vad_time": 6000,
                "end_vad_time": 5000,
                "audio_type": "pcm",
                "chinese2digital": 1,
                "punctuation": 2,
            },
            "business_info": "{}"
        }
        ws.send(json.dumps(start_msg))

        # 启动麦克风音频发送线程
        threading.Thread(target=self._send_mic_audio, args=(ws,), daemon=True).start()

    def _send_mic_audio(self, ws):
        """从麦克风实时读取音频并发送，带本地静音检测"""
        import pyaudio

        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000
        CHUNK = 1280

        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

        print("[ASR] 🎤 麦克风已开启，请开始说话... (静音 2.5 秒后自动结束)")

        # ----- 静音检测参数 -----
        SILENCE_THRESHOLD = 300   # 16-bit PCM 静音阈值
        SILENCE_TIMEOUT = 2.5     # 静音持续 2.5 秒后结束

        silence_counter = 0.0
        frame_counter = 0
        ANALYSIS_FRAME_INTERVAL = int(16000 / 1280)  # 约 1 秒
        speaking = False

        try:
            while not self._stop_send:
                data = stream.read(CHUNK, exception_on_overflow=False)
                if not data:
                    continue

                # 计算当前帧音量
                audio_chunk = np.frombuffer(data, dtype=np.int16)
                rms = np.sqrt(np.mean(audio_chunk.astype(np.float32) ** 2))

                is_speech = rms > SILENCE_THRESHOLD

                # 缓存并发送数据
                with self.audio_data_lock:
                    self.audio_data.extend(data)
                ws.send(data, opcode=websocket.ABNF.OPCODE_BINARY)

                # 状态机：检测静音结束
                if is_speech:
                    if not speaking:
                        print("[ASR] 🗣️ 检测到语音，开始录音...")
                        speaking = True
                    # 如果正在播放 TTS（通过 self.is_playing 标志），触发打断
                    self.interrupt_event.set()
                    silence_counter = 0.0
                else:
                    if speaking:
                        silence_counter += 0.04  # 每帧 40ms
                        if silence_counter >= SILENCE_TIMEOUT:
                            print(f"[ASR] 🔇 检测到静音 {SILENCE_TIMEOUT} 秒，自动结束录音。")
                            self._stop_send = True
                            break

                # 后台实时分析
                frame_counter += 1
                if frame_counter >= ANALYSIS_FRAME_INTERVAL:
                    frame_counter = 0
                    threading.Thread(target=self._update_features, daemon=True).start()

                time.sleep(0.04)

        except Exception as e:
            print(f"[ASR] 发送异常: {e}")
        finally:
            try:
                ws.send(b'--end--', opcode=websocket.ABNF.OPCODE_BINARY)
                print("[ASR] 📤 已发送结束标志")
            except Exception as e:
                print(f"[ASR] 发送结束标志失败: {e}")
            stream.stop_stream()
            stream.close()
            p.terminate()


    def _update_features(self):
        """后台线程：持续更新当前录音的特征值（轻量级，不阻塞录音）"""
        with self.audio_data_lock:
            if len(self.audio_data) < 16000 * 2:  # 少于1秒的数据不分析
                return
            audio_np = np.frombuffer(self.audio_data, dtype=np.int16).copy()

        # 转成 float [-1, 1] 供计算
        audio_float = audio_np.astype(np.float32) / 32768.0

        try:
            # 过零率（轻量）
            zcr = np.mean(np.abs(np.diff(np.sign(audio_float))))
            # RMS 能量
            rms = np.sqrt(np.mean(audio_float**2))
            # 语速估算（过零率变化率）
            zcr_changes = np.diff(np.sign(audio_float))
            speech_rate = np.mean(np.abs(zcr_changes)) * 100

            self.current_features = {
                "过零率": round(float(zcr), 4),
                "RMS能量": round(float(rms), 6),
                "语速参考值": round(float(speech_rate), 2),
                "已录音时长(秒)": round(len(audio_float) / 16000, 1)
            }
            # 每10秒打印一次，避免刷屏
            if int(len(audio_float) / 16000) % 5 == 0:
                print(f"[实时分析] {self.current_features}")
        except Exception as e:
            pass  # 静默处理，不影响录音

    def recognize_from_mic(self):
        """识别麦克风实时语音（长语音）"""
        self.result_text = ""
        self.is_finished = False
        self._stop_send = False

        ws_url = self._build_handshake_url()
        print(f"[ASR] 🌐 连接地址: {ws_url}")

        self.ws = websocket.WebSocketApp(
            ws_url,
            header={"Authorization": f"Bearer {self.app_key}"},
            on_open=self.on_open,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close
        )




        self.ws.run_forever(ping_interval=10, ping_timeout=5)

        # ----- 等待服务端返回最后结果，最多等待 5 秒 -----
        wait_count = 0
        while not self.is_finished and wait_count < 50:  # 50 * 0.1 = 5 秒
            time.sleep(0.1)
            wait_count += 1
        if not self.is_finished:
            print("[ASR] ⚠️ 未收到最后一句结果，强制结束")
            if self.ws:
                self.ws.close()

        # ----- 录音结束，保存音频和提取特征 -----
        if len(self.audio_data) > 0:
            # 1. 将 PCM 数据转为 numpy 数组 (int16)
            audio_np = np.frombuffer(self.audio_data, dtype=np.int16)

            # 2. 保存为 WAV 文件
            # 2. 归一化音量后再保存
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            wav_filename = f"recording_{timestamp}.wav"

            # 计算最大绝对值，防止溢出
            max_val = np.max(np.abs(audio_np))
            if max_val > 0:
                # 归一化到 0.8 倍满量程（留点余量避免削波）
                audio_np_normalized = (audio_np / max_val * 0.8 * 32767).astype(np.int16)
            else:
                audio_np_normalized = audio_np

            sf.write(wav_filename, audio_np_normalized, 16000, subtype='PCM_16')

            self.audio_saved_path = wav_filename
            print(f"[ASR] 💾 音频已保存到: {wav_filename}")

            # 3. 提取声学特征（复用之前的函数）
            try:
                self.features = extract_audio_features(wav_filename)
                # 保存特征为 JSON
                feature_filename = f"features_{timestamp}.json"
                with open(feature_filename, 'w', encoding='utf-8') as f:
                    json.dump(self.features, f, ensure_ascii=False, indent=2)
                print(f"[ASR] 📊 特征已保存到: {feature_filename}")
            except Exception as e:
                print(f"[ASR] ❌ 特征提取失败: {e}")
                self.features = None
        else:
            print("[ASR] ⚠️ 没有录到音频数据")
        return self.result_text

class ReplyWorker(threading.Thread):
    """独立线程：从队列取文本 → 调用大模型 → TTS播放"""
    def __init__(self, text_queue, chat_history, app_key, interrupt_event):
        super().__init__(daemon=True)
        self.text_queue = text_queue
        self.chat_history = chat_history
        self.history_lock = threading.Lock()
        self.app_key = app_key
        self.running = True
        self.interrupt_event = interrupt_event

    def run(self):
        while self.running:
            try:
                user_text = self.text_queue.get(timeout=1)  # 阻塞等待
            except queue.Empty:
                continue
            if user_text is None:  # 退出信号
                break

            # 调用大模型（使用全局 call_llm）
            reply = call_llm(user_text, self.chat_history)
            print(f"🤖 助手: {reply}")

            # 合成并播放TTS
            tts_synthesize(reply, interrupt_event=self.interrupt_event)

            # 更新对话历史（加锁）
            with self.history_lock:
                self.chat_history.append({"role": "user", "content": user_text})
                self.chat_history.append({"role": "assistant", "content": reply})
                if len(self.chat_history) > 10:
                    self.chat_history = self.chat_history[-10:]

    def stop(self):
        self.running = False
        self.text_queue.put(None)  # 唤醒线程


if __name__ == "__main__":
    print("🚀 启动「脑长青」语音对话系统（双线程模式）")
    print("=" * 60)
    print("系统将持续监听，您说话后自动结束，同时后台处理回复。")
    print("按 Ctrl+C 退出程序。\n")

    client = VivoLongASRClient(APP_KEY)
    text_queue = queue.Queue()
    chat_history = []  # 共享对话历史

    # 启动处理线程
    worker = ReplyWorker(text_queue, chat_history, APP_KEY, client.interrupt_event)
    worker.start()

    # 启动录音循环（主线程）
    try:
        # 初次问候（由系统主动发起）
        greeting = "您好，我是脑长青助手"
        print(f"🤖 助手: {greeting}")
        tts_synthesize(greeting, interrupt_event=client.interrupt_event)
        with worker.history_lock:
            chat_history.append({"role": "assistant", "content": greeting})

        while True:
            print("\n🎤 请说话...（说完后静音自动结束）")
            user_text = client.recognize_from_mic()
            if not user_text.strip():
                print("⚠️ 未识别到有效语音，请再说一遍。")
                continue

            print(f"👤 用户: {user_text}")

            # 保存用户发言到文本文件
            with open("conversation_log.txt", "a", encoding="utf-8") as f:
                f.write(f"{user_text}\n")

            # 提取的特征已保存在 client.features 中，可后续使用
            if client.features and isinstance(client.features, dict):
                print(f"📊 声学特征: 语速={client.features.get('语速参考值')}, 基频={client.features.get('基频均值(Hz)')}")

            # 将文本放入队列，由后台线程处理
            text_queue.put(user_text)

    except KeyboardInterrupt:
        print("\n👋 对话结束，感谢使用。")
    finally:
        worker.stop()
        worker.join(timeout=2)