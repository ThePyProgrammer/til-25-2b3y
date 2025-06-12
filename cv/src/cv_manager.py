"""Manages the CV model."""

from typing import Any
from mmdet.apis import init_detector, inference_detector
from mmdet.utils import get_test_pipeline_cfg
from mmdet.utils.large_image import merge_results_by_nms
from mmengine.dataset import pseudo_collate
from mmcv.transforms import Compose
from sahi.slicing import slice_image
import mmcv
import torch

BATCH_SIZE = 4
RET_THRES = 0.8

class CVManager:
    def __init__(self):
        # This is where you can initialize your model and any static
        # configurations.
        self.model = init_detector(
            config="mmdetection/configs/til-ai/curr_conf.py",
            checkpoint="model.pth",
        )
        test_pipeline = get_test_pipeline_cfg(self.model.cfg.copy())
        test_pipeline[0].type = "mmdet.LoadImageFromNDArray"
        self.test_pipeline = Compose(test_pipeline)

    def cv(self, image_batch: list[bytes]) -> list[list[dict[str, Any]]]:
        """Performs object detection on an image.

        Args:
            image: The image file in bytes.

        Returns:
            A list of `dict`s containing your CV model's predictions. See
            `cv/README.md` for the expected format.
        """
        with torch.no_grad():
            data = pseudo_collate(
                list(
                    map(
                        self.test_pipeline,
                        ({"img": img} for img in map(mmcv.imfrombytes, image_batch)),
                    )
                )
            )
            
            with torch.autocast(device_type="cuda"):
                results = self.model.test_step(data)

            overall_pred_bboxes = torch.tensor([], dtype=torch.half, device="cuda")
            overall_pred_labels = torch.tensor([], dtype=torch.uint8, device="cuda")
            overall_pred_imgs = torch.tensor([], dtype=torch.uint8, device="cuda")
            for i, preds in enumerate(results):
                instances = preds.pred_instances
                mask = (instances.scores >= RET_THRES)
                pred_labels = instances.labels[mask]
                pred_bboxes = instances.bboxes[mask]
                pred_bboxes[:, 2] = pred_bboxes[:, 2] - pred_bboxes[:, 0]
                pred_bboxes[:, 3] = pred_bboxes[:, 3] - pred_bboxes[:, 1]
                pred_imgs = torch.full(pred_labels.size(), i, dtype=torch.uint8, device="cuda")
                
                overall_pred_bboxes = torch.cat((overall_pred_bboxes, pred_bboxes))
                overall_pred_labels =  torch.cat((overall_pred_labels, pred_labels))
                overall_pred_imgs = torch.cat((overall_pred_imgs, pred_imgs))
                
            overall_pred_bboxes = overall_pred_bboxes.tolist()
            overall_pred_labels = overall_pred_labels.tolist()
            overall_pred_imgs = overall_pred_imgs.tolist()
            
            curr_pred = []
            last_img_num = 0
            predictions = []
            for i, bbox, label in zip(overall_pred_imgs, overall_pred_bboxes, overall_pred_labels):
                while last_img_num < i:
                    predictions.append(curr_pred)
                    last_img_num += 1
                    curr_pred = []
                
                curr_pred.append({
                    "bbox": bbox,
                    "category_id": label
                })
                
            while len(predictions) < len(image_batch):
                predictions.append(curr_pred)
                last_img_num += 1
                curr_pred = []

            return predictions
    
    def cv_slice(self, image: bytes) -> list[dict[str, Any]]:
        """Performs object detection on an image.

        Args:
            image: The image file in bytes.

        Returns:
            A list of `dict`s containing your CV model's predictions. See
            `cv/README.md` for the expected format.
        """

        img = mmcv.imfrombytes(image)
        height, width = img.shape[:2]
        slices = slice_image(img, slice_width=1333, slice_height=800)
        cv_imgs = pseudo_collate(
            list(
                map(
                    self.test_pipeline,
                    ({"img": img} for img in slices.images),
                )
            )
        )

        with torch.autocast(device_type="cuda"):
            results = self.model.test_step(cv_imgs)
            
        # check if the results are empty
        is_empty = True
        for result in results:
            if len(result.pred_instances.scores):
                is_empty = False
                break
        if is_empty:
            return []

        image_result = merge_results_by_nms(
            results,
            slices.starting_pixels,
            src_image_shape=(height, width),
            nms_cfg={"type": "nms", "iou_threshold": 0.25},
        )

        predictions = []
        for label, score, (x1, y1, x2, y2) in zip(
            image_result.pred_instances.labels,
            image_result.pred_instances.scores,
            image_result.pred_instances.bboxes,
        ):
            print(score)
            if score < RET_THRES:
                continue
            predictions.append(
                {
                    "bbox": list(map(lambda x: x.item(), [x1, y1, x2 - x1, y2 - y1])),
                    "category_id": label.item(),
                }
            )

        return predictions

