import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/v8/DAMI-yolov8l.yaml')
    # model.load('yolov8n.pt') # loading pretrain weights
    model.train(data='/root/autodl-tmp/damiyolo/dataset/pest24.yaml',
                cache=False,
                imgsz=1280,
                epochs=500,
                batch=2,
                close_mosaic=10,
                workers=8,
                device='0',
                optimizer='SGD', # using SGD
                # resume='', # last.pt path
                # amp=False, # close amp
                # fraction=0.2,
                project='runs/train',
                name='dami',
                )
