from ultralytics import YOLO

model = YOLO("yolo11s-cls.pt")

if __name__ == '__main__':
    # Train the model
    results = model.train(data=r'd:\datasets\LSPD\Images_YOLO_cls',
                        epochs=100,
                        imgsz=416,
                        patience=10,
                        batch=64,
                        workers=8,
                        optimizer='AdamW',
                        cos_lr=True,
                        lr0=1e-4,
                        lrf=0.05,
                        device=0,
                        project='nsfw_cls_models_11s',
                        name='train_cls_run_rc4',
                        save_period=1,
                        plots=True,
                        cache=None,
                        profile=True,
                        verbose=True)
    # Evaluate model perf on validation set
    results = model.val()