# -*- coding: utf-8 -*-
"""
对话管理器：封装实时对话、特征提取、大模型回复、TTS 播放、评估报告生成及清理。
供前端（如 PyQt）调用，实现完整的“脑长青”认知评估流程。
"""

import os
import glob
import shutil
import threading
import time
from datetime import datetime

# 导入你已有的模块
from main_1 import VivoLongASRClient, APP_KEY
from llm_tts import call_llm, tts_synthesize
from extract_features import extract_audio_features
from cognitive_assessment import run_evaluation


class DialogueManager:
    """
    对话管理器
    状态：空闲 → 对话中 → 录音中 → 停止录音（自动处理回复）→ 继续录音或结束对话
    """

    def __init__(self, app_key=APP_KEY):
        self.app_key = app_key
        self.client = None
        self.chat_history = []
        self.is_dialogue_active = False
        self.is_recording = False
        self.current_user_text = ""   # 最新一轮识别文本
        self.current_reply = ""       # 最新一轮回复

    # ==================== 前端调用的核心接口 ====================

    def start_dialogue(self):
        """开始新对话（重置所有状态）"""
        if self.is_dialogue_active:
            print("[对话] 已有对话进行中，请先结束当前对话")
            return
        self.client = VivoLongASRClient(self.app_key)
        self.chat_history = []
        self.is_dialogue_active = True
        self.is_recording = False
        self.current_user_text = ""
        self.current_reply = ""
        print("[对话] ✅ 已开始，等待录音...")
        # 可在这里添加首次问候（通过 TTS）
        # tts_synthesize("您好，我是脑长青助手，请点击说话按钮开始。")

    def start_recording(self):
        """点击“说话”按钮：开始录音"""
        if not self.is_dialogue_active:
            raise RuntimeError("请先调用 start_dialogue() 开始对话")
        if self.is_recording:
            print("[录音] 已在录音中，请勿重复点击")
            return
        self.is_recording = True
        self.client.start_recording()
        print("[录音] 🎤 开始录音...")

    def stop_recording(self):
        """
        点击“结束说话”按钮：停止录音，返回识别文本，并触发后台处理（LLM+TTS）。
        返回值：(user_text, reply)
        """
        if not self.is_recording:
            print("[录音] 未在录音状态，请先点击“说话”")
            return None, None

        self.is_recording = False
        # 停止录音并获得识别文本
        user_text = self.client.stop_recording()
        self.current_user_text = user_text

        if not user_text or not user_text.strip():
            print("[录音] ⚠️ 未识别到有效语音")
            return user_text, None

        # ---------- 并行处理（特征提取 + LLM + TTS） ----------
        # 1. 后台线程提取声学特征（不阻塞主流程）
        if self.client.audio_saved_path and os.path.exists(self.client.audio_saved_path):
            threading.Thread(
                target=extract_audio_features,
                args=(self.client.audio_saved_path,),
                daemon=True
            ).start()

        # 2. 主线程调用大模型和 TTS（可放在后台线程避免阻塞界面，但此处为了简化，先同步执行）
        #    若希望界面不卡顿，可将以下部分放在另一个线程。
        reply = call_llm(user_text, self.chat_history)
        self.current_reply = reply
        print(f"[对话] 🤖 助手: {reply}")

        # 3. TTS 合成并播放（播放期间可检测打断，但需要传递 interrupt_event）
        tts_synthesize(reply, interrupt_event=self.client.interrupt_event)

        # 4. 更新对话历史
        self.chat_history.append({"role": "user", "content": user_text})
        self.chat_history.append({"role": "assistant", "content": reply})
        if len(self.chat_history) > 10:
            self.chat_history = self.chat_history[-10:]

        # 5. 归档音频和文本到 input/（供评估系统使用）
        self._archive_to_input(user_text)

        return user_text, reply

    def end_dialogue(self):
        """
        点击“结束对话”按钮：终止对话，触发综合评估，生成报告，并清理临时文件。
        """
        if not self.is_dialogue_active:
            print("[对话] 当前无活跃对话")
            return

        # 如果正在录音，先停止
        if self.is_recording:
            self.stop_recording()

        # 标记对话结束
        self.is_dialogue_active = False
        print("[对话] 正在结束对话，生成评估报告...")

        # 调用评估模块（它会读取 input/ 下所有文件）
        try:
            run_evaluation("input", "output")
            print("[对话] ✅ 评估报告已生成，保存于 output/ 目录")
        except Exception as e:
            print(f"[对话] ❌ 评估过程出错: {e}")

        # 清理 input/ 目录（删除所有 .wav 和 .txt）
        self._clean_input_dir()
        print("[对话] 🧹 临时文件已清理")

        # 重置客户端（释放资源）
        if self.client and self.client.ws:
            try:
                self.client.ws.close()
            except:
                pass
        self.client = None

    # ==================== 内部辅助方法 ====================

    def _archive_to_input(self, text):
        """将当前录音文件和 ASR 文本复制到 input/ 目录"""
        if not self.client or not self.client.audio_saved_path:
            return
        input_dir = "input"
        os.makedirs(input_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(self.client.audio_saved_path))[0]
        # 复制音频
        src_wav = self.client.audio_saved_path
        dst_wav = os.path.join(input_dir, f"{base_name}.wav")
        shutil.copy2(src_wav, dst_wav)
        # 保存文本
        txt_path = os.path.join(input_dir, f"{base_name}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"[归档] 已保存 {dst_wav} 和 {txt_path}")

    def _clean_input_dir(self):
        """清空 input/ 下的所有 .wav 和 .txt 文件"""
        patterns = ["*.wav", "*.txt"]
        for pattern in patterns:
            for f in glob.glob(os.path.join("input", pattern)):
                try:
                    os.remove(f)
                    print(f"[清理] 已删除 {f}")
                except Exception as e:
                    print(f"[清理] 删除 {f} 失败: {e}")


# ==================== 独立测试（模拟前端调用） ====================
if __name__ == "__main__":
    print("=== 测试 DialogueManager ===")
    dm = DialogueManager()

    # 模拟前端按钮点击
    dm.start_dialogue()
    input("按 Enter 模拟点击“说话”按钮...")
    dm.start_recording()
    input("按 Enter 模拟点击“结束说话”按钮...")
    user_text, reply = dm.stop_recording()
    print(f"用户: {user_text}")
    print(f"助手: {reply}")

    # 模拟多轮（可重复调用 start_recording / stop_recording）
    # 这里不再重复，直接结束对话
    input("按 Enter 结束对话并生成报告...")
    dm.end_dialogue()
    print("测试完成。")