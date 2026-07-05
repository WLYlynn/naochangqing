import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os

# ===== 修复中文显示 =====
matplotlib.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Arial Unicode MS",
]
matplotlib.rcParams["axes.unicode_minus"] = False
# ========================


def extract_audio_features(audio_file, output_dir="result", plot=True):
    """
    提取音频特征，并可选择生成可视化图表。

    参数:
        audio_file (str): 音频文件路径（支持 WAV, MP3 等）
        output_dir (str): 图表输出目录，默认 "result"
        plot (bool): 是否生成图表，默认 True

    返回:
        dict: 包含各项声学特征的字典
    """
    # 确保输出目录存在
    if plot:
        os.makedirs(output_dir, exist_ok=True)

    # ========== 读取音频 ==========
    y, sr = librosa.load(audio_file, sr=None)  # sr=None 保持原始采样率
    print(f"采样率: {sr} Hz, 时长: {len(y) / sr:.2f} 秒, 样本数: {len(y)}")

    # ========== 1. 时域波形图 ==========
    if plot:
        plt.figure(figsize=(12, 4))
        librosa.display.waveshow(y, sr=sr, alpha=0.8)
        plt.title("波形图 (时域)")
        plt.xlabel("时间 (秒)")
        plt.ylabel("振幅")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "waveform.png"), dpi=150)
        plt.close()

    # ========== 2. 频谱图 (STFT) ==========
    if plot:
        D = librosa.stft(y)
        D_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        plt.figure(figsize=(12, 5))
        librosa.display.specshow(D_db, sr=sr, x_axis="time", y_axis="hz")
        plt.colorbar(format="%+2.0f dB")
        plt.title("频谱图 (STFT)")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "spectrogram.png"), dpi=150)
        plt.close()

    # ========== 3. 梅尔频谱图 ==========
    if plot:
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        plt.figure(figsize=(12, 5))
        librosa.display.specshow(mel_db, sr=sr, x_axis="time", y_axis="mel", fmax=8000)
        plt.colorbar(format="%+2.0f dB")
        plt.title("梅尔频谱图")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "mel_spectrogram.png"), dpi=150)
        plt.close()

    # ========== 4. MFCC ==========
    if plot:
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        plt.figure(figsize=(12, 5))
        librosa.display.specshow(mfccs, sr=sr, x_axis="time")
        plt.colorbar()
        plt.title("MFCC (前13维)")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "mfcc.png"), dpi=150)
        plt.close()

    # ========== 5. 统计特征（数值） ==========
    # 过零率
    zero_crossings = librosa.feature.zero_crossing_rate(y)
    zcr_mean = np.mean(zero_crossings)
    print(f"过零率 (均值): {zcr_mean:.4f}")

    # 均方根能量
    rms = librosa.feature.rms(y=y)
    rms_mean = np.mean(rms)
    print(f"RMS能量 (均值): {rms_mean:.6f}")

    # 基频 (使用pYIN算法)
    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=80, fmax=450, sr=sr)
    f0_mean = np.nanmean(f0)
    f0_std = np.nanstd(f0)
    print(f"基频 (均值, 仅有声段): {f0_mean:.2f} Hz")
    print(f"基频 (标准差): {f0_std:.2f} Hz")

    # 频谱质心
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
    sc_mean = np.mean(spectral_centroids)
    print(f"频谱质心 (均值): {sc_mean:.2f} Hz")

    # 语速估算（用过零率变化率作为简单指标）
    zcr_changes = np.diff(zero_crossings[0])
    speech_rate_estimate = np.mean(np.abs(zcr_changes)) * 100
    print(f"语速参考值: {speech_rate_estimate:.2f}")

    # ========== 构建返回字典 ==========
    features = {
        "过零率均值": round(float(zcr_mean), 4),
        "RMS能量均值": round(float(rms_mean), 6),
        "基频均值(Hz)": round(float(f0_mean), 2) if not np.isnan(f0_mean) else 0,
        "基频标准差(Hz)": round(float(f0_std), 2) if not np.isnan(f0_std) else 0,
        "频谱质心均值(Hz)": round(float(sc_mean), 2),
        "语速参考值": round(float(speech_rate_estimate), 2),
        "音频时长(秒)": round(len(y) / sr, 2),
    }

    if plot:
        print(
            "\n✅ 所有特征图已保存到 {} 文件夹，数值统计打印如上。".format(output_dir)
        )

    return features
# ========== 独立运行入口（保持原有功能） ==========
if __name__ == "__main__":
    # ========== 配置 ==========
    audio_file = "audio/opfirst.wav"  # 可以修改为你的文件
    output_dir = "result"
    # =========================

    # 调用函数，生成图表并打印数值
    features = extract_audio_features(audio_file, output_dir, plot=True)
    print("\n返回的特征字典:")
    print(features)
