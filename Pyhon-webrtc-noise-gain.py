import wave
import numpy as np
from webrtc_noise_gain import AudioProcessor

# 初始化处理器
auto_gain_dbfs = 3  # 0 表示不启用自动增益
noise_suppression_level = 4  # 噪声抑制等级，0~4
processor = AudioProcessor(auto_gain_dbfs, noise_suppression_level)

# 打开输入音频（必须是 16kHz, 16-bit, mono）
input_path = '3.wav'
output_path = '3_4_ouput.wav'

with wave.open(input_path, 'rb') as wf:
    assert wf.getframerate() == 16000, "必须是 16kHz 采样率"
    assert wf.getnchannels() == 1, "必须是单声道"
    assert wf.getsampwidth() == 2, "必须是16位采样（2字节）"
    
    audio_data = wf.readframes(wf.getnframes())

# 每帧处理：10ms = 160 samples = 320 bytes
frame_size = 320
processed_audio = bytearray()

for i in range(0, len(audio_data), frame_size):
    frame = audio_data[i:i+frame_size]
    if len(frame) < frame_size:
        frame = frame + b'\x00' * (frame_size - len(frame))  # padding
    result = processor.Process10ms(frame)
    processed_audio.extend(result.audio)

# 写入新音频文件
with wave.open(output_path, 'wb') as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)  # 16-bit
    wf.setframerate(16000)
    wf.writeframes(processed_audio)

print(f"处理完成，输出文件保存在: {output_path}")
