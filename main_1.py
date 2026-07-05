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
import pyaudio
import wave
import io
from datetime import datetime
from extract_features import extract_audio_features
from llm_tts import call_llm, tts_synthesize

# ================== 配置 ==================
APP_KEY = "sk-xuanji-2026525296-T1VZV2pHS3hJVXpZcXZOZg=="
TTS_ENGINE_ID = "short_audio_synthesis_jovi"
TTS_VCN = "vivoHelper"
TTS_SPEED = 50
TTS_VOLUME = 80
ENGINE_ID = "longasrlisten"
USER_ID = uuid.uuid4().hex[:32]
# =========================================

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
        self.audio_data = bytearray()          # 当前录音的PCM数据（用于保存）
        self.audio_saved_path = None
        self.features = None
        self.audio_data_lock = threading.Lock()
        self.current_features = None
        self.interrupt_event = threading.Event()   # 用于打断TTS

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
            elif action == "result":
                code = data.get("code", -1)
                result_data = data.get("data", {})
                if code == 8:
                    var_text = result_data.get("var", "")
                    if var_text:
                        print(f"[ASR] 📝 中间: {var_text}")
                elif code == 0:
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
                    self.ws.close()
            elif action == "error":
                print(f"[ASR] ❌ 错误: {data}")
                self.is_finished = True
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
        print("[ASR] 🎤 麦克风已开启，请开始说话...")
        try:
            while not self._stop_send:
                data = stream.read(CHUNK, exception_on_overflow=False)
                if data:
                    with self.audio_data_lock:
                        self.audio_data.extend(data)
                    ws.send(data, opcode=websocket.ABNF.OPCODE_BINARY)
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

    def start_recording(self):
        """由界面调用：开始录音，建立WebSocket连接，开启麦克风"""
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
        # 在独立线程中运行WebSocket，避免阻塞主线程
        threading.Thread(target=self.ws.run_forever, kwargs={"ping_interval": 10, "ping_timeout": 5}, daemon=True).start()

    def stop_recording(self):
        """由界面调用：停止录音，等待最终结果，返回识别文本和特征"""
        self._stop_send = True  # 通知音频线程停止
        # 等待服务端返回最终结果（最多等待10秒）
        wait_count = 0
        while not self.is_finished and wait_count < 100:  # 100 * 0.1 = 10秒
            time.sleep(0.1)
            wait_count += 1
        if not self.is_finished:
            print("[ASR] ⚠️ 未收到最后一句结果，强制结束")
            if self.ws:
                self.ws.close()

        # 保存音频和提取特征
        if len(self.audio_data) > 0:
            audio_np = np.frombuffer(self.audio_data, dtype=np.int16)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            wav_filename = f"recording_{timestamp}.wav"
            # 归一化
            max_val = np.max(np.abs(audio_np))
            if max_val > 0:
                audio_np_normalized = (audio_np / max_val * 0.8 * 32767).astype(np.int16)
            else:
                audio_np_normalized = audio_np
            sf.write(wav_filename, audio_np_normalized, 16000, subtype="PCM_16")
            self.audio_saved_path = wav_filename
            print(f"[ASR] 💾 音频已保存到: {wav_filename}")
        else:
            print("[ASR] ⚠️ 没有录到音频数据")
        return self.result_text


# ========== 主程序（连续对话） ==========
if __name__ == "__main__":
    print("🚀 启动「脑长青」语音对话系统（连续对话模式）")
    print("在每次录音前，程序会等待您按 Enter 开始录音，再按 Enter 结束录音。")
    print("输入 'quit' 或按 Ctrl+C 退出。\n")

    client = VivoLongASRClient(APP_KEY)
    chat_history = []  # 保存多轮对话历史

    while True:
        try:
            cmd = input("按 Enter 开始录音（或输入 quit 退出）...")
            if cmd.strip().lower() == "quit":
                break

            # 开始录音
            client.start_recording()
            print("\n🎤 录音中... 请说话。说完后按 Enter 结束录音。")
            input("按 Enter 结束录音...")

            text = client.stop_recording()
            print(f"\n📝 识别结果: {text}")

            # ---------- 归档到 input/ 目录（供评估系统使用） ----------
            import shutil

            input_dir = "input"
            os.makedirs(input_dir, exist_ok=True)

            if client.audio_saved_path and os.path.exists(client.audio_saved_path):
                # 基础名（不含扩展名）
                base_name = os.path.splitext(os.path.basename(client.audio_saved_path))[
                    0
                ]

                # 复制音频到 input/
                dest_wav = os.path.join(input_dir, f"{base_name}.wav")
                shutil.copy2(client.audio_saved_path, dest_wav)
                print(f"[归档] 音频已复制到 {dest_wav}")

                # 保存 ASR 文本到 input/
                txt_path = os.path.join(input_dir, f"{base_name}.txt")
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(text)
                print(f"[归档] 文本已保存到 {txt_path}")
            else:
                print("[归档] 未找到音频文件，跳过归档")
            # ----------------------------------------------------------

            if not text.strip():
                print("⚠️ 未识别到有效语音，请重试。")
                continue

            # === 并行处理 ===
            # 1. 在后台线程中提取声学特征（不阻塞主流程）
            if client.audio_saved_path:
                threading.Thread(
                    target=extract_audio_features,
                    args=(client.audio_saved_path,),
                    daemon=True,
                ).start()

            # 2. 主线程继续调用大模型和TTS
            reply = call_llm(text, chat_history)
            print(f"🤖 助手: {reply}")
            tts_synthesize(reply, interrupt_event=client.interrupt_event)

            # 更新对话历史
            chat_history.append({"role": "user", "content": text})
            chat_history.append({"role": "assistant", "content": reply})
            if len(chat_history) > 10:
                chat_history = chat_history[-10:]

        except KeyboardInterrupt:
            # 在退出前等待所有后台线程完成（可选）
            for thread in threading.enumerate():
                if thread != threading.main_thread() and thread.is_alive():
                    thread.join(timeout=1)
            break
        except Exception as e:
            print(f"❌ 发生错误: {e}")
            continue