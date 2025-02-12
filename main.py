from components.model_trainer import SpeechTrainer

def main():
    trainer = SpeechTrainer(config_path="./configs/config.yaml") 
    trainer.run()  # 執行完整的訓練流程

if __name__ == "__main__":
    main()
