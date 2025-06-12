import inspect
from mmcv.transforms import BaseTransform
from mmdet.structures.bbox import HorizontalBoxes
from mmpretrain.registry import TRANSFORMS
import mmengine
from typing import Dict, List, Optional
from pprint import pprint

import numpy
import torch

try:
    import albumentations
except ImportError:
    albumentations = None


# 'Albu' is used in previous versions of mmpretrain, here is for compatibility
# users can use both 'Albumentations' and 'Albu'.
@TRANSFORMS.register_module(["AlbuHBBoxes"])
class AlbuHBBoxes(BaseTransform):
    """Wrapper to use augmentation from albumentations library.

    **Required Keys:**

    - img

    **Modified Keys:**

    - img
    - img_shape

    Adds custom transformations from albumentations library.
    More details can be found in
    `Albumentations <https://albumentations.readthedocs.io>`_.
    An example of ``transforms`` is as followed:

    .. code-block::

        [
            dict(
                type='ShiftScaleRotate',
                shift_limit=0.0625,
                scale_limit=0.0,
                rotate_limit=0,
                interpolation=1,
                p=0.5),
            dict(
                type='RandomBrightnessContrast',
                brightness_limit=[0.1, 0.3],
                contrast_limit=[0.1, 0.3],
                p=0.2),
            dict(type='ChannelShuffle', p=0.1),
            dict(
                type='OneOf',
                transforms=[
                    dict(type='Blur', blur_limit=3, p=1.0),
                    dict(type='MedianBlur', blur_limit=3, p=1.0)
                ],
                p=0.1),
        ]

    Args:
        transforms (List[Dict]): List of albumentations transform configs.
        keymap (Optional[Dict]): Mapping of mmpretrain to albumentations
            fields, in format {'input key':'albumentation-style key'}.
            Defaults to None.

    Example:
        >>> import mmcv
        >>> from mmpretrain.datasets import Albumentations
        >>> transforms = [
        ...     dict(
        ...         type='ShiftScaleRotate',
        ...         shift_limit=0.0625,
        ...         scale_limit=0.0,
        ...         rotate_limit=0,
        ...         interpolation=1,
        ...         p=0.5),
        ...     dict(
        ...         type='RandomBrightnessContrast',
        ...         brightness_limit=[0.1, 0.3],
        ...         contrast_limit=[0.1, 0.3],
        ...         p=0.2),
        ...     dict(type='ChannelShuffle', p=0.1),
        ...     dict(
        ...         type='OneOf',
        ...         transforms=[
        ...             dict(type='Blur', blur_limit=3, p=1.0),
        ...             dict(type='MedianBlur', blur_limit=3, p=1.0)
        ...         ],
        ...         p=0.1),
        ... ]
        >>> albu = Albumentations(transforms)
        >>> data = {'img': mmcv.imread('./demo/demo.JPEG')}
        >>> data = albu(data)
        >>> print(data['img'].shape)
        (375, 500, 3)
    """

    def __init__(self, transforms: List[Dict], keymap: Optional[Dict] = None):
        if albumentations is None:
            raise RuntimeError("albumentations is not installed")
        else:
            from albumentations import (
                Compose as albu_Compose,
                BboxParams as albu_BboxParams,
            )

        assert isinstance(transforms, list), "transforms must be a list."
        if keymap is not None:
            assert isinstance(keymap, dict), "keymap must be None or a dict. "

        self.transforms = transforms

        self.aug = albu_Compose(
            [self.albu_builder(t) for t in self.transforms],
            bbox_params=albu_BboxParams(
                format="pascal_voc", label_fields=["bbox_labels"], clip=True
            ),
        )

        if not keymap:
            self.keymap_to_albu = dict(
                img="image",
                gt_bboxes_labels="bbox_labels",
            )
        else:
            self.keymap_to_albu = keymap
        self.keymap_back = {v: k for k, v in self.keymap_to_albu.items()}

    def albu_builder(self, cfg: Dict):
        """Import a module from albumentations.

        It inherits some of :func:`build_from_cfg` logic.
        Args:
            cfg (dict): Config dict. It should at least contain the key "type".
        Returns:
            obj: The constructed object.
        """

        assert isinstance(cfg, dict) and "type" in cfg, (
            "each item in " "transforms must be a dict with keyword 'type'."
        )
        args = cfg.copy()

        obj_type = args.pop("type")
        if mmengine.is_str(obj_type):
            obj_cls = getattr(albumentations, obj_type)
        elif inspect.isclass(obj_type):
            obj_cls = obj_type
        else:
            raise TypeError(
                f"type must be a str or valid type, but got {type(obj_type)}"
            )

        if "transforms" in args:
            args["transforms"] = [
                self.albu_builder(transform) for transform in args["transforms"]
            ]

        return obj_cls(**args)

    @staticmethod
    def mapper(d, keymap):
        """Dictionary mapper.

        Renames keys according to keymap provided.
        Args:
            d (dict): old dict
            keymap (dict): {'old_key':'new_key'}
        Returns:
            dict: new dict.
        """

        updated_dict = {}
        for k, v in zip(d.keys(), d.values()):
            new_k = keymap.get(k, k)
            updated_dict[new_k] = d[k]
        return updated_dict

    def transform(self, results: Dict) -> Dict:
        """Transform function to perform albumentations transforms.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Transformed results, 'img' and 'img_shape' keys are
                updated in result dict.
        """
        assert "img" in results, "No `img` field in the input."
        assert "gt_bboxes" in results, "No bboxes in the input."
        assert isinstance(results["gt_bboxes"], HorizontalBoxes), "Wrong bbox type."

        # dict to albumentations format
        results = self.mapper(results, self.keymap_to_albu)

        # map HorizontalBBox to albu format
        results["bboxes"] = results["gt_bboxes"].tensor.numpy()

        results = self.aug(**results)

        # back to the original format
        results = self.mapper(results, self.keymap_back)
        if type(results["gt_bboxes_labels"]) == list:
            # normally means it's empty
            results["gt_bboxes_labels"] = numpy.array(
                results["gt_bboxes_labels"], dtype=int
            )
        results["gt_bboxes"] = HorizontalBoxes(results["bboxes"], dtype=torch.float32)
        results["gt_ignore_flags"] = numpy.array(
            [False] * len(results["bboxes"]), dtype=bool
        )
        results["img_shape"] = results["img"].shape[:2]

        return results

    def __repr__(self):
        """Print the basic information of the transform.

        Returns:
            str: Formatted string.
        """
        repr_str = self.__class__.__name__
        repr_str += f"(transforms={repr(self.transforms)})"
        return repr_str
