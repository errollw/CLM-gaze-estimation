import os
import cv2

out = cv2.VideoWriter('output.avi', -1, 20.0, (800, 600))

imgs_dir = "vid_imgs"
fns = [os.path.join(imgs_dir,fn) for fn in os.listdir(imgs_dir) if fn.endswith(".jpg")]
fns.sort(key=lambda x: os.path.getmtime(x))

for i, fn in enumerate(fns):

    print "processing frame %d"%i
    frame = cv2.imread(fn)
    out.write(frame)

out.release()
print "done"