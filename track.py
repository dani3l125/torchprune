import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import cv2
import torchprune as tp
import torchprune.util.models.deeplab as dmodels
import time
import argparse
import numpy as np
import cv2 as cv


parser = argparse.ArgumentParser(description='This sample demonstrates Lucas-Kanade Optical Flow calculation. \
                                              The example file can be downloaded from: \
                                              https://www.bogotobogo.com/python/OpenCV_Python/images/mean_shift_tracking/slow_traffic_small.mp4')
parser.add_argument('image', type=str, help='path to image file')
args = parser.parse_args()

cap = cv.VideoCapture(args.image)

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = np.random.randint(0, 255, (100, 3))

# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

while(1):
    ret, frame = cap.read()
    if not ret:
        print('No frames grabbed!')
        break

    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # calculate optical flow
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Select good points
    if p1 is not None:
        good_new = p1[st==1]
        good_old = p0[st==1]

    # draw the tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
        frame = cv.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
    img = cv.add(frame, mask)

    cv.imshow('frame', img)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

cv.destroyAllWindows()



# parser = argparse.ArgumentParser(description='Process some integers.')
# parser.add_argument('--file',
#                     default='alds12',
#                     help='model file')
# parser.add_argument('--static',
#                     default=False,
#                     help='Using static image')
# parser.add_argument('--orig',
#                     default=1,
#                     help='pytorch original model')
# parser.add_argument('--ratio',
#                     type=float,
#                     default=-1,
#                     help='manual keep ratio')
# parser.add_argument('--compare_origs',
#                     default=False,
#                     help='compare backbones')
# args = parser.parse_args()
#
# if __name__ == "__main__":
#     # Initialize semantic segmentation model
#     if args.compare_origs:
#         models = [
#                   # tp.ALDSNetPlus(tp.util.net.NetHandle(dmodels.deeplabv3_resnet50(21).eval(), "deeplabv3_resnet50"), None, None),
#                   #tp.ALDSNetPlus(tp.util.net.NetHandle(dmodels.deeplabv3_resnet50_new(21).eval(), "deeplabv3_resnet50_new"), None, None),
#                   tp.ALDSNetPlus(tp.util.net.NetHandle(dmodels.deeplabv3_mobilenet_v3_large(21).eval(), "deeplabv3_mobilenet_v3_large"), None, None)
#                   ]
#     elif not args.orig:
#         model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True).eval()
#         if args.ratio == -1:
#             tp.util.train.load_checkpoint(args.file, model, loc='cpu')
#         model.eval()
#     else:
#         model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True).eval()
#
#     # Capture camera frame
#     if not args.static:
#         print("=====================")
#         print("Using camera")
#         cam = cv2.VideoCapture(0)  # set the port of the camera as before
#         retval, input_image = cam.read()  # return a True bolean and and the image if all go right
#         input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
#         plt.imshow(input_image)
#         cam.release()  # Closes video file or capturing device.
#     else:
#         print("=====================")
#         print("Not using camera")
#         input_image = Image.open('deeplab1.png')
#         input_image = input_image.convert("RGB")
#
#     # Prepare model input
#     preprocess = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ])
#
#     input_tensor = preprocess(input_image)
#     input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model
#
#     # Manual compression if needed
#     if args.ratio != -1:
#         print("=====================")
#         print(f"Starting compression with ratio{args.ratio}")
#         with torch.no_grad():
#             output = model(input_batch)[0]
#         model.compress(keep_ratio=args.ratio)
#         torch.save(model.state_dict(), 'alds10')
#     if args.orig == 1:
#         torch.onnx.export(model,  # model being run
#         input_batch,  # model input (or a tuple for multiple inputs)
#         "super_resolution2.onnx",  # where to save the model (can be a file or file-like object)
#         export_params = True,  # store the trained parameter weights inside the model file
#                         opset_version = 10,  # the ONNX version to export the model to
#                                         do_constant_folding = True,  # whether to execute constant folding for optimization
#                                                               input_names = ['input'],  # the model's input names
#                                                                             output_names = [
#                                                                                                'output'],  # the model's output names
#                                                                                            dynamic_axes = {
#             'input': {0: 'batch_size'},  # variable length axes
#             'output': {0: 'batch_size'}})
#
#     # Inference
#     if args.compare_origs:
#         for model in models:
#             start = time.time()
#             with torch.no_grad():
#                 if not args.orig:
#                     output = model(input_batch)[0]
#                 else:
#                     output = model(input_batch)['out'][0]
#                 print("Inference took {}sec".format(time.time() - start))
#     else:
#         start = time.time()
#         with torch.no_grad():
#             if not args.orig:
#                 output = model(input_batch)[0]
#             else:
#                 output = model(input_batch)['out'][0]
#             print("Inference took {}sec".format(time.time()-start))
#         output_predictions = output.argmax(0)
#
#         # create a color pallette, selecting a color for each class
#         palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
#         colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
#         colors = (colors % 255).numpy().astype("uint8")
#
#         # plot the semantic segmentation predictions of 21 classes in each color
#         r = Image.fromarray(output_predictions.byte().cpu().numpy())
#         r.putpalette(colors)
#         if not args.static:
#             plt.figure()
#             plt.imshow(r)
#             plt.show()

