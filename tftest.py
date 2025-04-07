import tensorflow as tf

# 檢查 TensorFlow 版本
print("TensorFlow 版本:", tf.__version__)

# 檢查 GPU 是否可用
gpu_devices = tf.config.list_physical_devices('GPU')
if gpu_devices:
    print(f"可用的 GPU 數量: {len(gpu_devices)}")
    for device in gpu_devices:
        print("偵測到的 GPU:", device)
else:
    print("未偵測到 GPU，請確認 CUDA 和 cuDNN 是否安裝正確！")