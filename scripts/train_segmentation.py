from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n-seg.yaml")  # build a new model from YAML

if __name__ == '__main__':
    # Train the model
    results = model.train(data='hmmm\dataset_imgs_segmentation.yaml',
                        epochs=250,
                        imgsz=640,
                        patience=25,
                        batch=16,
                        device=0,
                        project='nsfw_segmentation_models_11n',
                        name='train_segment_run_1',
                        save_period=20,
                        plots=True,
                        cache=None,
                        verbose=True)
    # Evaluate model perf on validation set
    results = model.val()