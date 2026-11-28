import cv2, numpy as np
from pathlib import Path
from slopewatch.fusion import load_segmenter

palette = np.array([[222,223,211],[36,116,167],[177,81,36],[31,133,95],[137,111,171]], np.uint8)
seg = load_segmenter()
image_path = Path("data/segmentation/images/0007.png")
mask = cv2.imread(str(Path("data/segmentation/masks/0007.png")), cv2.IMREAD_GRAYSCALE)
pred = seg(image_path)
img = cv2.imread(str(image_path))
def colour(m): return palette[np.clip(m,0,4)][:,:,::-1]
panels = [img, colour(mask), colour(pred)]
labels = ["input", "ground truth", "prediction"]
canvas = np.full((img.shape[0]+46, img.shape[1]*3+24, 3), 255, np.uint8)
for i,(panel,label) in enumerate(zip(panels,labels)):
    x = i*(img.shape[1]+8)
    canvas[40:40+panel.shape[0], x:x+panel.shape[1]] = panel
    cv2.putText(canvas, label, (x+8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30,40,45), 2, cv2.LINE_AA)
cv2.imwrite("docs/segmentation_demo.png", canvas)
acc = (pred==mask).mean()
print("pixel acc", round(float(acc),3))
