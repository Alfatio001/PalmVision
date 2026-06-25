from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")

    model.train(
        data="data.yaml",

        epochs=50,

        imgsz=512,
        batch=2,

        device=0,
        workers=2,

        cache=False,
        amp=True,

        save=True,
        save_period=-1,
        plots=False,

        optimizer="auto",

        project="runs",
        name="palm_yolov8n",
        exist_ok=True
    )

if __name__ == "__main__":
    main()