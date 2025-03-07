import os
import math
from utils.data_utils import DataUtils
from utils.preprocess import AudioPreprocess
import pandas as pd
import h5py
import json
from model.model import TransformerCTC, CustomSchedule
import tensorflow as tf
import matplotlib.pyplot as plt

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

def main():
    # Data preparation
    dataset_folder = "./dataset"
    data_utils = DataUtils(dataset_folder, limit=30000)
    train_df, valid_df, test_df = data_utils.load_data()
    
    # Create vocabulary
    sentences = pd.concat([train_df['sentence'], valid_df['sentence'], test_df['sentence']])
    word2idx = data_utils.create_vocab(sentences, './vocab.json')
    
    # Set up audio processing
    audio_folder = os.path.join(dataset_folder, "clips")
    audio_processor = AudioPreprocess(target_sr=16000, save_folder='./features', n_fft=1024, hop_length=256, win_length=1024)
    
    # Process audio files
    data_utils.batch_process_to_hdf5(audio_processor, audio_folder, train_df, './features/train_spectrograms.h5', workers=4)
    data_utils.batch_process_to_hdf5(audio_processor, audio_folder, valid_df, './features/valid_spectrograms.h5', workers=4)
    
    # Find max sentence length
    max_length, longest_sentence = data_utils.find_max_sentence_length(sentences, word2idx)
    print(f"\n最長 token 序列長度: {max_length}\n對應句子: {longest_sentence}")
    
    # Prepare datasets
    audio_input_shape = (513, 238)  # Adjust based on your spectrogram shape
    target_seq_len = 30  # Max sequence length for targets
    
    train_dataset = audio_processor.create_chunked_dataset(
        './features/train_spectrograms.h5', 
        train_df, 
        data_utils, 
        word2idx, 
        target_seq_len, 
        audio_input_shape, 
        batch_size=32, 
        shuffle=True
    )
    
    valid_dataset = audio_processor.create_chunked_dataset(
        './features/valid_spectrograms.h5', 
        valid_df, 
        data_utils, 
        word2idx, 
        target_seq_len, 
        audio_input_shape, 
        batch_size=32, 
        shuffle=False
    )
    
    # Initialize model
    model = TransformerCTC(
        audio_input_shape=audio_input_shape,
        vocab_size=len(word2idx),
        d_model=128,
        num_enc_layers=3,
        num_heads=4,
        dff=256,
        dropout_rate=0.3
    )
    
    # Setup optimizer and learning rate
    learning_rate = CustomSchedule(128)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    
    # Define custom CTC loss function
    def ctc_loss_fn(y_true, y_pred):
        return model.compute_ctc_loss(y_true, y_pred)
    
    # Compile model
    model.compile(
        optimizer=optimizer,
        loss=ctc_loss_fn
    )
    
    # Build model and print summary
    model.build(input_shape=(None, *audio_input_shape))
    model.summary()
    
    # Training parameters
    steps_per_epoch = math.ceil(len(train_df) / 32)
    validation_steps = math.ceil(len(valid_df) / 32)
    
    # Optional: Enable mixed precision for faster training
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    
    # Create directories for checkpoints
    os.makedirs('checkpoints', exist_ok=True)
    
    print("\n開始訓練模型...")
    
    # Train model
    history = model.fit(
        train_dataset,
        validation_data=valid_dataset,
        epochs=10,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                'checkpoints/best_model.ckpt',
                save_best_only=True,
                monitor='val_loss',
                save_weights_only=True
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir='./logs',
                update_freq='epoch'
            )
        ]
    )
    
    # Plot training history
    if 'accuracy' in history.history and 'val_accuracy' in history.history:
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Loss')
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Accuracy')
        plt.show()
    else:
        plt.figure(figsize=(8, 4))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Loss')
        plt.show()

if __name__ == "__main__":
    main()