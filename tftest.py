import tensorflow as tf

<<<<<<< HEAD
=======
<<<<<<< HEAD
>>>>>>> aee2d04 (update)
# 檢查 TensorFlow 版本
print("TensorFlow 版本:", tf.__version__)

# 檢查 GPU 是否可用
gpu_devices = tf.config.list_physical_devices('GPU')
if gpu_devices:
    print(f"可用的 GPU 數量: {len(gpu_devices)}")
    for device in gpu_devices:
        print("偵測到的 GPU:", device)
else:
<<<<<<< HEAD
    print("未偵測到 GPU，請確認 CUDA 和 cuDNN 是否安裝正確！")
=======
    print("未偵測到 GPU，請確認 CUDA 和 cuDNN 是否安裝正確！")
=======
# 列出所有可用的 GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    print(f"找到 {len(gpus)} 個 GPU：")
    for gpu in gpus:
        print(gpu)
else:
    print("未找到 GPU，TensorFlow 可能正在使用 CPU。")
>>>>>>> 8819392 (初始化專案)
>>>>>>> aee2d04 (update)
