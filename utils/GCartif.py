# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
# # #!/usr/bin/env python3
# import tensorflow as tf
# import cv2
# import numpy as np
# import os
# import csv
# from tqdm import tqdm
# from scipy.stats import skew
# from scipy.ndimage import center_of_mass
# import sys

# # Ensure eval_NN (with crop_to_bbox) is importable
# sys.path.insert(0, os.path.abspath('..'))
# from eval_NN import crop_to_bbox

# # Paths
# MODEL_PATH = (
#     '/DATASERVER/MIC/GENERAL/STUDENTS/amartic/mscthesis/'
#     'polypclassificationmi/code/data/snapshots/all/'
#     'hypVSadn_HDall2023_efficientnet_0_regularized0.0_256x256_'
#     '1in_nf64_bnTrue_fcdo0.0_balancedTrue_loss_fl_gamma1.0_sgd_5fold0_best.h5'
# )
# SEQUENCES_FN = (
#     '/DATASERVER/MIC/GENERAL/STUDENTS/amartic/mscthesis/'
#     'polypclassificationmi/data/degraded_image_list.txt'
# )
# OUTPUT_ROOT = (
#     '/DATASERVER/MIC/GENERAL/STUDENTS/amartic/mscthesis/'
#     'polypclassificationmi/results/gradcam_artif'
# )

# class HeatmapCreator:
#     def __init__(self, model_path, sequences_fn,
#                  output_root=OUTPUT_ROOT, classes=[0,1],
#                  smooth_samples=0, smooth_sigma=0.1):
#         """
#         model_path: path to Keras .h5 model
#         sequences_fn: txt file with lines: degraded_full_path ann_full_path label pid
#         output_root: directory to save side-by-side images and CSV
#         classes: classes to evaluate
#         smooth_samples: number of SmoothGrad samples (>1 to enable)
#         smooth_sigma: noise sigma for SmoothGrad
#         """
#         # Load model
#         model = tf.keras.models.load_model(model_path, compile=False)
#         model = self._flatten_functional(model)
#         conv_layer = self._find_last_conv_layer(model)
#         self.model = tf.keras.Model(
#             inputs=model.input,
#             outputs=[model.get_layer(conv_layer).output, model.output]
#         )
#         self.input_size = tuple(model.input.shape.as_list()[1:3])
#         self.sequences_fn = sequences_fn
#         self.output_root = output_root
#         self.classes = classes
#         self.smooth_samples = smooth_samples
#         self.smooth_sigma = smooth_sigma
#         os.makedirs(self.output_root, exist_ok=True)

#     def _flatten_functional(self, model):
#         cfg = model.get_config()
#         layers = cfg['layers']
#         flat = []
#         weight_map = {}
#         while layers:
#             L = layers.pop(0)
#             if L['class_name'] == 'Functional':
#                 inner = L['config']['layers']
#                 for il in inner:
#                     weight_map[il['name']] = L['name']
#                 inner = inner[1:]
#                 inner[0]['inbound_nodes'] = L['inbound_nodes']
#                 outb = L['config']['output_layers'][0]
#                 outb.append({})
#                 layers[0]['inbound_nodes'] = [[outb,],]
#                 layers = inner + layers
#             else:
#                 flat.append(L)
#         cfg['layers'] = flat
#         flat_model = tf.keras.Model().from_config(cfg)
#         for lyr in flat_model.layers:
#             name = lyr.name
#             if name in weight_map:
#                 orig = model.get_layer(weight_map[name])
#                 path = [name]
#                 while weight_map.get(path[0]):
#                     path.insert(0, weight_map[path[0]])
#                 for p in path[1:]:
#                     orig = orig.get_layer(p)
#                 lyr.set_weights(orig.get_weights())
#             else:
#                 lyr.set_weights(model.get_layer(name).get_weights())
#         return flat_model

#     def _find_last_conv_layer(self, model):
#         for layer in reversed(model.layers):
#             if len(layer.output_shape) == 4:
#                 return layer.name
#         raise ValueError("No 4D conv layer found")

#     def _preprocess(self, img_path, ann_path):
#         img = cv2.imread(img_path)
#         ann = cv2.imread(ann_path, 0)
#         ann = cv2.resize(ann, (img.shape[1], img.shape[0]))
#         crop_resized = crop_to_bbox(img, ann, padding=0.2, output_size=self.input_size)
#         contours, _ = cv2.findContours(ann, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         x, y, w, h = cv2.boundingRect(contours[0])
#         orig_crop = img[y:y+h, x:x+w]
#         tensor = tf.expand_dims(crop_resized.astype(np.float32), axis=0)
#         return tensor, orig_crop, (x, y, w, h)

#     def create_frames(self):
#         # Read all entries, expecting 4 columns per line:
#         # <degraded_full_path> <annotation_full_path> <label> <pid>
#         with open(self.sequences_fn, 'r') as f:
#             raw_entries = [line.strip() for line in f if line.strip()]

#         entries = []
#         for line in raw_entries:
#             tokens = line.split()
#             if len(tokens) != 4:
#                 print(f"[!] Skipping malformed line (expected 4 tokens): {line}")
#                 continue

#             degraded_full_path, ann_full_path, label, pid = tokens

#             # Extract directory and filename for the degraded image
#             img_dir  = os.path.dirname(degraded_full_path)
#             img_file = os.path.basename(degraded_full_path)

#             # Extract directory and filename for the annotation
#             ann_dir  = os.path.dirname(ann_full_path)
#             ann_file = os.path.basename(ann_full_path)

#             # Store as tuple matching the original signature
#             entries.append((img_dir, img_file, ann_dir, ann_file, label, pid))

#         results = []
#         thr = 0.5

#         for img_dir, img_file, ann_dir, ann_file, label, pid in tqdm(entries):
#             img_path = os.path.join(img_dir, img_file)
#             ann_path = os.path.join(ann_dir, ann_file)
#             # Preprocess
#             tensor, orig_crop, (x0, y0, w0, h0) = self._preprocess(img_path, ann_path)

#             # Compute (Smooth)Grad-CAM
#             if self.smooth_samples > 1:
#                 cam_acc = None
#                 for _ in range(self.smooth_samples):
#                     noise = tf.random.normal(tf.shape(tensor), stddev=self.smooth_sigma)
#                     with tf.GradientTape() as tape:
#                         convs, preds = self.model(tensor + noise)
#                         cls = int(tf.argmax(preds[0]))
#                         loss = preds[:, cls]
#                     grads = tape.gradient(loss, convs)
#                     guided = tf.cast(convs>0, tf.float32) * tf.cast(grads>0, tf.float32) * grads
#                     weights = tf.reduce_mean(guided, axis=(1,2))
#                     cam_i = tf.reduce_sum(weights * convs[0], axis=-1).numpy()
#                     cam_i = np.maximum(cam_i, 0)
#                     cam_acc = cam_i if cam_acc is None else cam_acc + cam_i
#                 cam = cam_acc / float(self.smooth_samples)
#             else:
#                 with tf.GradientTape() as tape:
#                     convs, preds = self.model(tensor)
#                     cls = int(tf.argmax(preds[0]))
#                     loss = preds[:, cls]
#                 grads = tape.gradient(loss, convs)
#                 guided = tf.cast(convs>0, tf.float32) * tf.cast(grads>0, tf.float32) * grads
#                 weights = tf.reduce_mean(guided, axis=(1,2))
#                 cam = tf.reduce_sum(weights * convs[0], axis=-1).numpy()

#             # Normalize
#             cam = np.maximum(cam, 0)
#             cam /= (cam.max() + 1e-8)

#             # Heatmap overlay
#             heat = (cam * 255).astype(np.uint8)
#             heat_resized = cv2.resize(heat, (w0, h0))
#             heat_color = cv2.applyColorMap(heat_resized, cv2.COLORMAP_VIRIDIS)

#             full_img = cv2.imread(img_path)
#             full_with_heat = full_img.copy()
#             region = full_with_heat[y0:y0+h0, x0:x0+w0]
#             overlay = cv2.addWeighted(region, 0.6, heat_color, 0.4, 0)
#             full_with_heat[y0:y0+h0, x0:x0+w0] = overlay

#             # Ground-truth overlay
#             ann_full = cv2.imread(ann_path, 0)
#             ann_full = cv2.resize(ann_full, (full_img.shape[1], full_img.shape[0]))
#             red = np.zeros_like(full_img); red[...,2] = 255
#             alpha = (ann_full > 0).astype(np.float32) * 0.4
#             full_with_gt = full_img.copy()
#             for c in range(3):
#                 full_with_gt[..., c] = (
#                     full_with_gt[..., c] * (1 - alpha) + red[..., c] * alpha
#                 ).astype(np.uint8)

#             # Stitch side by side and save
#             combined = np.concatenate([full_with_heat, full_with_gt], axis=1)
#             seq = os.path.basename(os.path.normpath(img_dir))
#             out_dir = os.path.join(self.output_root, seq)
#             os.makedirs(out_dir, exist_ok=True)
#             out_png = os.path.join(out_dir, f"{seq}_{img_file}_sidebyside.png")
#             cv2.imwrite(out_png, combined)

#             # Metrics
#             pred_prob = float(tf.reduce_max(preds[0]))
#             gt = int(label)
#             correct = int(cls == gt)

#             hF, wF = full_with_gt.shape[:2]
#             cx, cy = wF/2.0, hF/2.0
#             dy, dx = center_of_mass((heat_color[...,1] > 0).astype(np.float32))
#             dist_center = np.sqrt((dx-cx)**2 + (dy-cy)**2) / np.sqrt(cx**2 + cy**2)

#             heat_bin = np.zeros_like(ann_full, dtype=np.uint8)
#             heat_bin[y0:y0+h0, x0:x0+w0] = (
#                 heat_resized >= int(255 * thr)
#             ).astype(np.uint8)
#             inter = np.logical_and(heat_bin > 0, ann_full > 0).sum()
#             union = np.logical_or(heat_bin > 0, ann_full > 0).sum()
#             iou = float(inter) / union if union > 0 else 0.0

#             gray_crop = cv2.cvtColor(orig_crop, cv2.COLOR_BGR2GRAY)
#             lap_var = float(cv2.Laplacian(gray_crop, cv2.CV_64F).var())
#             sk_val = float(skew(gray_crop.ravel()))

#             pos_cm = center_of_mass(heat_bin.astype(np.float32))
#             neg_cm = center_of_mass((1 - heat_bin.astype(np.float32)))

#             results.append({
#                 'sequence': seq,
#                 'image': img_file,
#                 'predicted_class': cls,
#                 'probability': pred_prob,
#                 'ground_truth': gt,
#                 'correct': correct,
#                 'iou': iou,
#                 'laplacian_var': lap_var,
#                 'skewness': sk_val,
#                 'com_pos_x': pos_cm[1],
#                 'com_pos_y': pos_cm[0],
#                 'dist_center': dist_center,
#                 'com_neg_x': neg_cm[1],
#                 'com_neg_y': neg_cm[0],
#             })

#         # Write CSV
#         csv_path = os.path.join(self.output_root, 'metrics.csv')
#         if results:
#             with open(csv_path, 'w', newline='') as cf:
#                 fieldnames = list(results[0].keys())
#                 writer = csv.DictWriter(cf, fieldnames=fieldnames)
#                 writer.writeheader()
#                 for r in results:
#                     writer.writerow(r)

#         print(f"Saved side-by-side images to: {self.output_root}")
#         print(f"Saved CSV to: {csv_path}")

# if __name__ == '__main__':
#     hc = HeatmapCreator(
#         model_path=MODEL_PATH,
#         sequences_fn=SEQUENCES_FN,
#         output_root=OUTPUT_ROOT,
#         classes=[0,1],
#         smooth_samples=10,
#         smooth_sigma=0.1
#     )
#     hc.create_frames()

# #!/usr/bin/env python3
import tensorflow as tf
import cv2
import numpy as np
import os
import csv
from tqdm import tqdm
from scipy.stats import skew
from scipy.ndimage import center_of_mass
import sys

# Paths (as provided)
MODEL_PATH = (
    '/DATASERVER/MIC/GENERAL/STUDENTS/amartic/mscthesis/polypclassificationmi/code/utils/'
    'final_fold0.h5'
)
SEQUENCES_FN = (
    '/DATASERVER/MIC/GENERAL/STUDENTS/amartic/mscthesis/polypclassificationmi/data/oxford_experiment_artif_expanded.txt'
)
OUTPUT_ROOT = (
    '/DATASERVER/MIC/GENERAL/STUDENTS/amartic/mscthesis/polypclassificationmi/'
    'results/gradcam_artif_oxford_exp_01_07'
)

class HeatmapCreator:
    def __init__(self,
                 model_path,
                 sequences_fn,
                 output_root=OUTPUT_ROOT,
                 classes=[0, 1],
                 smooth_samples=0,
                 smooth_sigma=0.1):
        """
        model_path: path to Keras .h5 model
        sequences_fn: txt file with lines: degraded_full_path annotation_full_path label [pid]
        output_root: directory to save side-by-side images and CSV
        classes: classes to evaluate
        smooth_samples: number of SmoothGrad samples (>1 to enable)
        smooth_sigma: noise sigma for SmoothGrad
        """
        # Load the model without compilation
        model = tf.keras.models.load_model(model_path, compile=False)
        model = self._flatten_functional(model)
        # Identify the last convolutional layer
        conv_layer = self._find_last_conv_layer(model)
        # Build a model that outputs feature maps from the last conv layer and the final predictions
        self.model = tf.keras.Model(
            inputs=model.input,
            outputs=[model.get_layer(conv_layer).output, model.output]
        )
        # Expected input size for the model (height, width)
        self.input_size = tuple(model.input.shape.as_list()[1:3])
        self.sequences_fn = sequences_fn
        self.output_root = output_root
        self.classes = classes
        self.smooth_samples = smooth_samples
        self.smooth_sigma = smooth_sigma
        os.makedirs(self.output_root, exist_ok=True)

    def _flatten_functional(self, model):
        """
        Flatten nested Functional models if needed, to simplify layer access.
        """
        cfg = model.get_config()
        layers = cfg['layers']
        flat = []
        weight_map = {}
        while layers:
            L = layers.pop(0)
            if L['class_name'] == 'Functional':
                inner = L['config']['layers']
                for il in inner:
                    weight_map[il['name']] = L['name']
                # Rewire inbound nodes
                inner = inner[1:]
                inner[0]['inbound_nodes'] = L['inbound_nodes']
                outb = L['config']['output_layers'][0]
                outb.append({})
                layers[0]['inbound_nodes'] = [[outb,],]
                layers = inner + layers
            else:
                flat.append(L)
        cfg['layers'] = flat
        flat_model = tf.keras.Model().from_config(cfg)
        # Copy weights from original model
        for lyr in flat_model.layers:
            name = lyr.name
            if name in weight_map:
                orig = model.get_layer(weight_map[name])
                path = [name]
                while weight_map.get(path[0]):
                    path.insert(0, weight_map[path[0]])
                for p in path[1:]:
                    orig = orig.get_layer(p)
                lyr.set_weights(orig.get_weights())
            else:
                lyr.set_weights(model.get_layer(name).get_weights())
        return flat_model

    def _find_last_conv_layer(self, model):
        """
        Find the last convolutional layer (4D output) in the model.
        """
        for layer in reversed(model.layers):
            if len(layer.output_shape) == 4:
                return layer.name
        raise ValueError("No 4D convolutional layer found in the model.")

    def _preprocess(self, img_path, ann_path):
        """
        Read image and annotation, resize for model, and prepare mask.
        Returns:
          - tensor: resized image tensor for model input (1, H, W, 3)
          - full_img: original image for overlay
          - mask_full: binary mask of ground truth at original size
        """
        full_img = cv2.imread(img_path)
        if full_img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        ann = cv2.imread(ann_path, 0)
        if ann is None:
            raise FileNotFoundError(f"Annotation not found: {ann_path}")
        # Resize image for model input
        resized = cv2.resize(full_img,
                             (self.input_size[1], self.input_size[0]),
                             interpolation=cv2.INTER_LINEAR)
        tensor = tf.expand_dims(resized.astype(np.float32), axis=0)
        # Resize annotation mask to original image size
        ann_full = cv2.resize(ann,
                              (full_img.shape[1], full_img.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
        mask_full = np.logical_or(ann_full == 0, ann_full == 255).astype(np.uint8)
        return tensor, full_img, mask_full

    def create_frames(self):
        """
        Process sequences, generate Grad-CAM or SmoothGrad-CAM heatmaps,
        overlay ground truth, save images and CSV metrics.
        Only the first 17000 entries will be processed.
        """
        # Read and parse sequences file
        with open(self.sequences_fn, 'r') as f:
            raw_entries = [line.strip() for line in f if line.strip()]

        entries = []
        for line in raw_entries:
            tokens = line.split()
            if len(tokens) == 3:
                degraded_full_path, ann_full_path, label = tokens
                pid = None
            elif len(tokens) == 4:
                degraded_full_path, ann_full_path, label, pid = tokens
            else:
                print(f"[!] Skipping malformed line: {line}")
                continue

            img_dir, img_file = os.path.dirname(degraded_full_path), os.path.basename(degraded_full_path)
            ann_dir, ann_file = os.path.dirname(ann_full_path), os.path.basename(ann_full_path)
            entries.append((img_dir, img_file, ann_dir, ann_file, label, pid))

        if not entries:
            print(f"[!] No valid entries found in {self.sequences_fn}")
            return

        # Limit processing to the first 17000 entries
        entries = entries[:17000]

        results = []
        thr = 0.5

        # Iterate over entries with progress bar
        for img_dir, img_file, ann_dir, ann_file, label, pid in tqdm(entries,
                                                                     desc="Processing first 17000 entries"):
            img_path = os.path.join(img_dir, img_file)
            ann_path = os.path.join(ann_dir, ann_file)

            try:
                tensor, full_img, mask_full = self._preprocess(img_path, ann_path)
            except Exception as e:
                print(f"[!] Preprocessing error for {img_path} / {ann_path}: {e}")
                continue

            # Generate Grad-CAM or SmoothGrad-CAM
            if self.smooth_samples > 1:
                cam_acc = None
                for _ in range(self.smooth_samples):
                    noise = tf.random.normal(tf.shape(tensor), stddev=self.smooth_sigma)
                    with tf.GradientTape() as tape:
                        convs, preds = self.model(tensor + noise)
                        cls = int(tf.argmax(preds[0]))
                        loss = preds[:, cls]
                    grads = tape.gradient(loss, convs)
                    guided = tf.cast(convs > 0, tf.float32) * tf.cast(grads > 0, tf.float32) * grads
                    weights = tf.reduce_mean(guided, axis=(1, 2))
                    cam_i = tf.reduce_sum(weights * convs[0], axis=-1).numpy()
                    cam_i = np.maximum(cam_i, 0)
                    cam_acc = cam_i if cam_acc is None else cam_acc + cam_i
                cam = cam_acc / float(self.smooth_samples)
            else:
                with tf.GradientTape() as tape:
                    convs, preds = self.model(tensor)
                    cls = int(tf.argmax(preds[0]))
                    loss = preds[:, cls]
                grads = tape.gradient(loss, convs)
                guided = tf.cast(convs > 0, tf.float32) * tf.cast(grads > 0, tf.float32) * grads
                weights = tf.reduce_mean(guided, axis=(1, 2))
                cam = tf.reduce_sum(weights * convs[0], axis=-1).numpy()

            # Normalize and resize CAM for overlay
            cam = np.maximum(cam, 0)
            cam /= (cam.max() + 1e-8)
            heat = (cam * 255).astype(np.uint8)
            heat_resized = cv2.resize(heat,
                                       (self.input_size[1], self.input_size[0]),
                                       interpolation=cv2.INTER_LINEAR)
            full_h, full_w = full_img.shape[:2]
            heat_full = cv2.resize(heat_resized, (full_w, full_h), interpolation=cv2.INTER_LINEAR)
            heat_color_full = cv2.applyColorMap(heat_full, cv2.COLORMAP_VIRIDIS)

            # Overlay heatmap and ground truth mask
            overlay = cv2.addWeighted(full_img, 0.6, heat_color_full, 0.4, 0)
            red_layer = np.zeros_like(full_img); red_layer[..., 2] = 255
            alpha_mask = mask_full.astype(np.float32) * 0.4
            full_with_gt = full_img.copy()
            for c in range(3):
                full_with_gt[..., c] = (full_with_gt[..., c] * (1 - alpha_mask) + red_layer[..., c] * alpha_mask).astype(np.uint8)

            # Save side-by-side image
            seq = os.path.splitext(ann_file)[0]
            out_dir = os.path.join(self.output_root, seq)
            os.makedirs(out_dir, exist_ok=True)
            out_png = os.path.join(out_dir, f"{seq}_{img_file}_sidebyside.png")
            combined = np.concatenate([overlay, full_with_gt], axis=1)
            cv2.imwrite(out_png, combined)

            # Compute metrics
            pred_prob = float(tf.reduce_max(preds[0]))
            gt = int(label)
            correct = int(cls == gt)
            heat_bin = (heat_full >= int(255 * thr)).astype(np.uint8)
            inter = np.logical_and(heat_bin > 0, mask_full > 0).sum()
            union = np.logical_or(heat_bin > 0, mask_full > 0).sum()
            iou = float(inter) / union if union > 0 else 0.0
            try:
                dy, dx = center_of_mass(heat_bin.astype(np.float32))
                dist_center = np.sqrt((dx - full_w/2)**2 + (dy - full_h/2)**2) / np.sqrt((full_w/2)**2 + (full_h/2)**2)
            except:
                dist_center = None

            # Sharpness and skewness within bounding box
            try:
                contours, _ = cv2.findContours(mask_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    x, y, w, h = cv2.boundingRect(contours[0])
                    crop_gray = cv2.cvtColor(full_img[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
                    lap_var = float(cv2.Laplacian(crop_gray, cv2.CV_64F).var())
                    sk_val = float(skew(crop_gray.ravel()))
                else:
                    lap_var = sk_val = None
            except:
                lap_var = sk_val = None

            # Centers of mass for positive and negative regions
            try:
                pos = center_of_mass(heat_bin.astype(np.float32))
                neg = center_of_mass((1 - heat_bin.astype(np.float32)))
                com_pos_x, com_pos_y = pos[1], pos[0]
                com_neg_x, com_neg_y = neg[1], neg[0]
            except:
                com_pos_x = com_pos_y = com_neg_x = com_neg_y = None

            results.append({
                'sequence': seq,
                'image': img_file,
                'predicted_class': cls,
                'probability': pred_prob,
                'ground_truth': gt,
                'correct': correct,
                'iou': iou,
                'laplacian_var': lap_var,
                'skewness': sk_val,
                'com_pos_x': com_pos_x,
                'com_pos_y': com_pos_y,
                'dist_center': dist_center,
                'com_neg_x': com_neg_x,
                'com_neg_y': com_neg_y,
            })

        # Save metrics to CSV
        csv_path = os.path.join(self.output_root, 'metrics.csv')
        if results:
            with open(csv_path, 'w', newline='') as cf:
                writer = csv.DictWriter(cf, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)

        print(f"Saved side-by-side images to: {self.output_root}")
        print(f"Saved CSV to: {csv_path}")

if __name__ == '__main__':
    hc = HeatmapCreator(
        model_path=MODEL_PATH,
        sequences_fn=SEQUENCES_FN,
        output_root=OUTPUT_ROOT,
        classes=[0, 1],
        smooth_samples=10,
        smooth_sigma=0.1
    )
    hc.create_frames()



