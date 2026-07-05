# -*- coding: utf-8 -*-
"""
认知风险评估模块（综合评估版）
功能：读取 input/ 下所有音频 → 分别提取特征 → 汇总特征和文字 → AI综合评估 → 生成一份总报告
输入：多个音频文件 + 对应的文字文件（可选）
输出：一份综合报告 + 每个音频的特征图
"""

import json
import os
import re
import glob
import numpy as np
from datetime import datetime
import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import requests
import uuid

# ==========================================
# 【配置区】
# ==========================================

AI_NAME = "小帮"

# 蓝心大模型配置
APP_KEY = "sk-xuanji-2026525296-T1VZV2pHS3hJVXpZcXZOZg=="
DOMAIN = "api-ai.vivo.com.cn"
CHAT_URI = "/v1/chat/completions"
CHAT_MODEL = "Volc-DeepSeek-V3.2"

USE_REAL_LLM = True   # True=真实模型，False=模拟回复

INPUT_DIR = "input"
OUTPUT_DIR = "output"
RESULT_DIR = "output/figures"

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# 默认文字（当找不到对应 txt 时使用）
DEFAULT_TEXT = """我昨天早上七点多起床，先去小区楼下散步了半个小时，
然后在门口的早餐店买了豆浆和包子。
上午在家看了一会儿电视，是关于养生的节目。
中午我自己做了西红柿鸡蛋面，味道还不错。
下午跟老邻居在楼下聊了会儿天，说起来孙子快放暑假了。
晚上儿子打电话来，说周末要回来看我。"""


# ==========================================
# 功能1：分析单个音频特征（返回特征字典 + 生成特征图）
# ==========================================

def analyze_audio(audio_path, base_name):
    """
    分析单个音频，返回特征字典，并生成以 base_name 命名的特征图
    """
    print(f"  🎵 正在分析音频：{os.path.basename(audio_path)}")

    y, sr = librosa.load(audio_path, sr=None)
    duration = len(y) / sr

    # 基频
    f0, voiced_flag, _ = librosa.pyin(
        y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C6'), sr=sr
    )
    f0_voiced = f0[voiced_flag]
    if len(f0_voiced) > 0:
        f0_mean = float(np.mean(f0_voiced))
        f0_std = float(np.std(f0_voiced))
        f0_range = float(np.max(f0_voiced) - np.min(f0_voiced))
    else:
        f0_mean = f0_std = f0_range = 0

    voiced_ratio = float(np.mean(voiced_flag))
    rms = librosa.feature.rms(y=y)[0]
    silence_ratio = float(np.mean(rms < 0.01))
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    zcr_mean = float(np.mean(zcr))
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    spec_cent_mean = float(np.mean(spec_cent))

    # 韵律评分
    prosody_score = 10
    if f0_std < 10: prosody_score -= 4
    elif f0_std < 20: prosody_score -= 2
    if silence_ratio > 0.5: prosody_score -= 3
    elif silence_ratio > 0.3: prosody_score -= 1
    if voiced_ratio < 0.3: prosody_score -= 2
    prosody_score = max(0, min(10, prosody_score))

    # ===== 生成特征图 =====
    # 波形图
    plt.figure(figsize=(10, 3))
    librosa.display.waveshow(y, sr=sr, color='#2E86AB', alpha=0.8)
    plt.title("Waveform", fontweight='bold')
    plt.xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, f"{base_name}_waveform.png"), dpi=100)
    plt.close()

    # 频谱图
    D = librosa.stft(y)
    D_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    plt.figure(figsize=(10, 4))
    img = librosa.display.specshow(D_db, sr=sr, x_axis='time', y_axis='hz', cmap='viridis')
    plt.colorbar(img, format='%+2.0f dB')
    plt.title("Spectrogram", fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, f"{base_name}_spectrogram.png"), dpi=100)
    plt.close()

    # 基频曲线
    plt.figure(figsize=(10, 3))
    times = librosa.times_like(f0, sr=sr)
    plt.plot(times, f0, color='#E63946', linewidth=2, alpha=0.8)
    plt.title("F0 Curve (Prosody)", fontweight='bold')
    plt.xlabel("Time (s)")
    plt.ylabel("Hz")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, f"{base_name}_f0_curve.png"), dpi=100)
    plt.close()

    # 梅尔频谱图
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    plt.figure(figsize=(10, 4))
    img = librosa.display.specshow(mel_db, sr=sr, x_axis='time', y_axis='mel', fmax=8000, cmap='magma')
    plt.colorbar(img, format='%+2.0f dB')
    plt.title("Mel Spectrogram", fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, f"{base_name}_mel_spectrogram.png"), dpi=100)
    plt.close()

    features = {
        "duration": duration,
        "sample_rate": sr,
        "f0_mean": f0_mean,
        "f0_std": f0_std,
        "f0_range": f0_range,
        "voiced_ratio": voiced_ratio,
        "silence_ratio": silence_ratio,
        "zcr_mean": zcr_mean,
        "spectral_centroid_mean": spec_cent_mean,
        "prosody_score": prosody_score
    }

    print(f"     ✅ 时长：{duration:.2f}秒 | 韵律评分：{prosody_score}/10")
    return features


# ==========================================
# 功能2：读取 input 下所有文本文件并汇总
# ==========================================

def load_all_texts():
    """读取 input 文件夹中所有 .txt 文件，拼接成完整文本"""
    all_txt_files = glob.glob(os.path.join(INPUT_DIR, "*.txt"))

    if not all_txt_files:
        print("  ⚠️  未找到任何文本文件，使用默认示例文本")
        return DEFAULT_TEXT

    combined = []
    for txt_path in sorted(all_txt_files):
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    combined.append(content)
            print(f"  📝 已读取：{os.path.basename(txt_path)}")
        except Exception as e:
            print(f"  ⚠️  读取 {os.path.basename(txt_path)} 失败：{e}")

    if not combined:
        return DEFAULT_TEXT

    return "。\n".join(combined)

# ==========================================
# 功能3：汇总所有音频的特征和文字
# ==========================================

def aggregate_features(all_features, all_durations):
    """
    汇总多个音频的特征，生成综合特征字典
    对数值特征按时长加权平均
    """
    total_duration = sum(all_durations)
    if total_duration == 0:
        total_duration = 0.001  # 防止除零

    # 需要汇总的数值特征列表
    numeric_keys = [
        'f0_mean', 'f0_std', 'f0_range', 'voiced_ratio',
        'silence_ratio', 'zcr_mean', 'spectral_centroid_mean', 'prosody_score'
    ]

    # 加权平均
    weighted_avg = {}
    for key in numeric_keys:
        weighted_sum = sum(feat[key] * dur for feat, dur in zip(all_features, all_durations))
        weighted_avg[key] = round(weighted_sum / total_duration, 2)

    # 汇总特征字典（保留时长、采样率等）
    agg = {
        "duration": round(total_duration, 2),
        "sample_rate": int(all_features[0]['sample_rate']),  # 假设所有采样率相同
        "f0_mean": weighted_avg['f0_mean'],
        "f0_std": weighted_avg['f0_std'],
        "f0_range": weighted_avg['f0_range'],
        "voiced_ratio": weighted_avg['voiced_ratio'],
        "silence_ratio": weighted_avg['silence_ratio'],
        "zcr_mean": weighted_avg['zcr_mean'],
        "spectral_centroid_mean": weighted_avg['spectral_centroid_mean'],
        "prosody_score": weighted_avg['prosody_score'],
    }
    return agg


# ==========================================
# 功能4：AI认知评估（同前，只需传入综合特征）
# ==========================================

def ai_assess(text, speech_rate, features):
    print(f"  🤖 {AI_NAME}正在综合评估认知风险...")

    system_prompt = f"""
你是一位拥有30年经验的神经科医生，叫{AI_NAME}，专门从事阿尔茨海默症早期筛查工作。

请结合以下综合语音特征评估老人的整体认知状态：

【综合语音特征参考】
- 总音频时长：{features['duration']}秒
- 综合语速：{speech_rate}字/秒（正常约2-3字/秒）
- 平均语调变化（f0标准差）：{features['f0_std']} Hz（正常约20-40Hz）
- 综合韵律评分：{features['prosody_score']}/10
- 平均停顿比例：{features['silence_ratio']}（越高停顿越多）
- 平均浊音比例：{features['voiced_ratio']}

【评估维度】（每项0-10分，10分最健康）
1. 词汇丰富度：用词是否多样
2. 逻辑连贯性：说话有没有条理
3. 信息密度：实际内容多不多
4. 记忆表现：有没有重复、遗忘
5. 语言流畅度：说话卡不卡
6. 定向能力：时间地点人物清不清楚
7. 韵律自然度：语调是否自然

【风险等级】
- 70-60分：低风险（绿色）
- 59-45分：中低风险（浅绿色）
- 44-30分：中风险（黄色）
- 29-15分：中高风险（橙色）
- 14分以下：高风险（红色）

【输出格式】严格用JSON返回，不要有其他文字：
{{
    "scores": {{
        "词汇丰富度": 分数,
        "逻辑连贯性": 分数,
        "信息密度": 分数,
        "记忆表现": 分数,
        "语言流畅度": 分数,
        "定向能力": 分数,
        "韵律自然度": 分数
    }},
    "total_score": 总分,
    "risk_level": "风险等级",
    "risk_color": "颜色",
    "detailed_analysis": "详细分析（150字左右）",
    "suggestion": "具体建议（80字左右）",
    "attention_points": ["关注点1", "关注点2"]
}}
"""

    result = _call_llm(system_prompt, f"老人说的话（综合）：\n{text}")

    try:
        clean = result.replace("```json", "").replace("```", "").strip()
        analysis = json.loads(clean)
        print(f"     ✅ 评估完成，总分：{analysis.get('total_score', '?')}分")
        return analysis
    except Exception as e:
        print(f"     ❌ 结果解析失败：{e}")
        return {"error": f"解析失败：{str(e)}", "raw_result": result}


def _call_llm(system_prompt, user_message):
    if not USE_REAL_LLM:
        return _mock_reply(system_prompt, user_message)

    url = f"https://{DOMAIN}{CHAT_URI}"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]
    data = {
        "model": CHAT_MODEL,
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": 1000,
        "requestId": str(uuid.uuid4())
    }
    headers = {
        "Authorization": f"Bearer {APP_KEY}",
        "Content-type": "application/json",
    }
    try:
        response = requests.post(url, headers=headers, json=data, timeout=60)
        if response.status_code == 200:
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"]
            else:
                return str(result)
        else:
            return f"接口错误：{response.status_code} - {response.text}"
    except Exception as e:
        return f"调用失败：{str(e)}"


def _mock_reply(system_prompt, user_message):
    if "认知" in system_prompt or "风险" in system_prompt or "筛查" in system_prompt:
        return '''
{
    "scores": {
        "词汇丰富度": 7,
        "逻辑连贯性": 8,
        "信息密度": 6,
        "记忆表现": 7,
        "语言流畅度": 8,
        "定向能力": 9,
        "韵律自然度": 7
    },
    "total_score": 52,
    "risk_level": "中低风险",
    "risk_color": "浅绿色",
    "detailed_analysis": "综合来看，老人语言表达整体较好，词汇使用较为丰富，逻辑清晰有条理。说话流畅度不错，停顿较少。时间地点人物定向能力良好，能够准确描述日常活动。语音韵律自然，音调有一定起伏。整体认知状态良好，但信息密度略低，建议继续保持社交活动和脑力锻炼。",
    "suggestion": "建议保持规律作息，多参与社交活动，适当进行阅读、下棋等脑力活动。饮食注意营养均衡，多吃鱼类、坚果等健脑食物。定期体检，持续关注认知健康。",
    "attention_points": ["信息密度略低，可鼓励多分享细节", "建议每3个月复查一次", "多参与社交活动"]
}
'''
    return "我是小帮，很高兴为您服务！"


# ==========================================
# 功能5：生成综合报告
# ==========================================

def generate_report(analysis, speech_rate, agg_features, combined_text, audio_list):
    """
    生成综合评估报告
    audio_list: 每个元素为 (filename, duration)
    """
    if "error" in analysis:
        return f"""
{'=' * 60}
  ❌ 评估失败
{'=' * 60}

错误信息：{analysis.get('error', '未知错误')}

原始返回：
{analysis.get('raw_result', '')}
"""

    scores = analysis["scores"]
    total = analysis["total_score"]
    risk = analysis["risk_level"]
    color = analysis["risk_color"]
    detail = analysis["detailed_analysis"]
    suggestion = analysis["suggestion"]
    attention = analysis["attention_points"]

    color_emoji = {"绿色": "🟢", "浅绿色": "🟢", "黄色": "🟡", "橙色": "🟠", "红色": "🔴"}
    emoji = color_emoji.get(color, "⚪")

    # 构建音频列表信息
    audio_info = "\n".join([f"    • {fname}（{dur:.2f}秒）" for fname, dur in audio_list])

    report = f"""
{'=' * 60}
  🧠 {AI_NAME}综合认知风险评估报告
{'=' * 60}

📅 评估时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
🎵 分析音频数量：{len(audio_list)} 个
📊 总时长：{agg_features['duration']} 秒

【音频列表】
{audio_info}

{'=' * 60}
  🎤 一、综合语音特征数据
{'=' * 60}

  基本信息：
    • 总时长：        {agg_features['duration']} 秒
    • 综合语速：      {speech_rate} 字/秒（正常2-3）

  语调特征（加权平均）：
    • 平均音调：      {agg_features['f0_mean']} Hz
    • 音调变化：      {agg_features['f0_std']} Hz（正常20-40）
    • 音调范围：      {agg_features['f0_range']} Hz
    • 浊音比例：      {agg_features['voiced_ratio'] * 100:.0f}%
    • 停顿比例：      {agg_features['silence_ratio'] * 100:.0f}%
    • 韵律评分：      {agg_features['prosody_score']}/10

  其他特征：
    • 过零率：        {agg_features['zcr_mean']}
    • 频谱质心：      {agg_features['spectral_centroid_mean']} Hz

{'=' * 60}
  📋 二、各维度得分（满分10分）
{'=' * 60}

"""
    for name, score in scores.items():
        bar = "█" * score + "░" * (10 - score)
        report += f"  {name}：{bar} {score}/10\n"

    report += f"""
{'=' * 60}
  🎯 三、综合评估结果
{'=' * 60}

  {emoji} 风险等级：{risk}（{color}）

  📊 总分：{total}/70

{'=' * 60}
  💬 四、详细分析
{'=' * 60}

  {detail}

{'=' * 60}
  💡 五、健康建议
{'=' * 60}

  {suggestion}

{'=' * 60}
  ⚠️  六、需要重点关注
{'=' * 60}

"""

    for i, point in enumerate(attention, 1):
        report += f"  {i}. {point}\n"

    report += f"""
{'=' * 60}
  📝 七、老人综合陈述（拼接文本）
{'=' * 60}

  {combined_text}

{'=' * 60}

  ⚠️  重要提示：本评估由{AI_NAME}基于多段语音特征进行综合初步筛查，
      仅供健康参考，不能替代专业医疗诊断。
      如有疑虑，请及时就医。

{'=' * 60}
"""
    return report


# ==========================================
# 主函数（综合处理）
# ==========================================

def load_text_for_audio(wav_path):
    pass


def run_evaluation(input_dir="input", output_dir="output"):
    """
    读取 input_dir 下所有 .wav 和 .txt，进行多文件汇总评估，
    生成综合报告到 output_dir
    """
    print()
    print("=" * 60)
    print(f"  🧠 {AI_NAME}认知风险评估系统（多文件汇总版）")
    print("  💙 基于vivo蓝心大模型")
    print("=" * 60)
    print()

    # 确保目录存在
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # 动态设置结果图目录
    result_dir = os.path.join(output_dir, "figures")
    os.makedirs(result_dir, exist_ok=True)

    # 获取所有 .wav 文件
    wav_files = glob.glob(os.path.join(input_dir, "*.wav"))
    if not wav_files:
        print(f"❌ 在 {input_dir}/ 下没有找到任何 .wav 文件，请先录音。")
        return

    print(f"🎵 找到 {len(wav_files)} 个音频文件")
    wav_files.sort()

    all_features = []
    all_durations = []
    all_texts = []
    audio_list = []

    for wav_path in wav_files:
        base_name = os.path.splitext(os.path.basename(wav_path))[0]
        print(f"\n--- 处理：{base_name} ---")

        # 分析音频（传入 result_dir 以便保存特征图）
        global RESULT_DIR
        original_result_dir = RESULT_DIR
        RESULT_DIR = result_dir

        features = analyze_audio(wav_path, base_name)

        # 恢复原值
        RESULT_DIR = original_result_dir

        all_features.append(features)
        all_durations.append(features["duration"])
        audio_list.append((os.path.basename(wav_path), features["duration"]))

        # 读取对应文本
        txt_path = os.path.join(input_dir, f"{base_name}.txt")
        if os.path.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8") as f:
                text = f.read().strip()
            print(f"  📝 已读取文本：{os.path.basename(txt_path)}")
        else:
            text = f"（用户第 {len(all_texts) + 1} 段发言）"
            print(f"  ⚠️  未找到对应的文本文件")
        all_texts.append(text)

    # 汇总特征
    print("\n📊 正在汇总所有音频特征...")
    agg_features = aggregate_features(all_features, all_durations)
    combined_text = "\n".join(all_texts)

    # 计算综合语速
    total_chars = len(re.findall(r"[\u4e00-\u9fa5]", combined_text))
    speech_rate = round(total_chars / agg_features["duration"], 2)
    print(f"🗣️ 综合语速：{speech_rate} 字/秒（总字数 {total_chars}，总时长 {agg_features['duration']}秒）")

    # AI评估
    analysis = ai_assess(combined_text, speech_rate, agg_features)
    print()

    # 生成报告
    print("📄 正在生成综合评估报告...")
    report = generate_report(analysis, speech_rate, agg_features, combined_text, audio_list)

    print(report)

    # 保存报告
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = os.path.join(output_dir, f"综合报告_{timestamp}.txt")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"💾 报告已保存到：{report_file}")
    print(f"🖼️  特征图已保存到：{result_dir}/")
    print()
    print("=" * 60)
    print("  ✅ 评估完成！")
    print("=" * 60)
    print()
# ==========================================
# 辅助：语速计算
# ==========================================
def get_latest_file(dir_path, pattern="*.wav"):
    """获取指定目录下最新修改的文件"""
    files = glob.glob(os.path.join(dir_path, pattern))
    if not files:
        return None
    # 按修改时间从旧到新排序，取最后一个
    files.sort(key=lambda x: os.path.getmtime(x))
    return files[-1]

def calc_speech_rate(text, duration):
    if duration <= 0 or not text:
        return 0
    chars = re.findall(r'[\u4e00-\u9fa5]', text)
    if not chars:
        return 0
    return round(len(chars) / duration, 2)


# ==========================================
# 运行
# ==========================================

if __name__ == "__main__":
    run_evaluation("input", "output")