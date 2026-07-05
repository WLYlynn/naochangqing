import librosa
import soundfile as sf
import os

# ===== 修改这里 =====
input_file = "fydz.mp3"          # ⚠️ 改成你实际的MP3文件名
output_file = "audio/opfirst.wav"  # 输出到audio文件夹
# ===================

# 确保audio文件夹存在
os.makedirs("audio", exist_ok=True)

print(f"📂 正在读取: {input_file}")

# 加载MP3，强制转为单声道，采样率统一为22050Hz
y, sr = librosa.load(input_file, sr=22050, mono=True)

print(f"✅ 加载成功！采样率: {sr}Hz, 时长: {len(y)/sr:.2f}秒")

# 保存为WAV（16bit PCM格式）
sf.write(output_file, y, sr, subtype='PCM_16')

print(f"💾 已保存至: {output_file}")