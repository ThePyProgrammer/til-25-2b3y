import base64
import json
import random
from collections import defaultdict, deque
from enum import StrEnum, auto
from pathlib import Path
from typing import TypedDict

import constants
import jiwer
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


class TaskType(StrEnum):
    ASR = auto()
    CV = auto()
    OCR = auto()

    def get_filename(self, index: int):
        match self:
            case TaskType.ASR:
                return f"sample_{index}.wav"
            case TaskType.CV:
                return f"images/{index}.jpg"
            case TaskType.OCR:
                return f"sample_{index}.jpg"

    def get_gt_path(self, index: int):
        match self:
            case TaskType.ASR:
                return f"sample_{index}.txt"
            case TaskType.CV:
                raise Exception("nah you can't do that for CV")
            case TaskType.OCR:
                return f"sample_{index}_text.txt"


class Task(TypedDict):
    type: TaskType
    index: int


wer_transforms = jiwer.Compose(
    [
        jiwer.ToLowerCase(),
        jiwer.SubstituteRegexes({"-": " "}),
        jiwer.RemovePunctuation(),
        jiwer.ReduceToListOfListOfWords(),
    ]
)

cer_transforms = jiwer.Compose(
    [
        jiwer.SubstituteRegexes({"-": ""}),
        jiwer.RemoveWhiteSpace(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.ReduceToListOfListOfChars(),
    ]
)


class COCOPatched(COCO):
    def __init__(self, annotations):
        # The varnames here are disgusting, but they're used by other
        # non-overridden methods so don't touch them.
        self.dataset, self.anns, self.cats, self.imgs = {}, {}, {}, {}
        self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)

        assert (
            type(annotations) == dict
        ), f"Annotation format {type(annotations)} not supported"
        print("Annotations loaded.")
        self.dataset = annotations
        self.createIndex()


class TaskHandler:
    def __init__(self, data_dir: Path, shuffle: bool = False):
        # filepath to load testcase data from
        self.data_dir = data_dir
        self.queue: deque[Task] = deque()
        self.shuffle = shuffle
        self.can_get_new = True

        with open(data_dir / "cv" / "annotations.json") as cv_anns_file:
            cv_anns_raw = json.load(cv_anns_file)

        self.cv_img_info = {}
        for img_info in cv_anns_raw["images"]:
            self.cv_img_info[img_info["id"]] = img_info

        self.cv_ann_info = {}
        for ann_info in cv_anns_raw["annotations"]:
            img_id = ann_info["image_id"]
            if img_id not in self.cv_ann_info:
                self.cv_ann_info[img_id] = []
            self.cv_ann_info[img_id].append(ann_info)

        self.cv_categories = cv_anns_raw["categories"]

        self.init_testcases()

    def get_cv_annotation(self, img_id):
        return {
            "images": [self.cv_img_info[img_id]],
            "annotations": self.cv_ann_info[img_id],
            "categories": self.cv_categories,
        }

    def reset(self):
        self.queue = deque()
        self.init_testcases()
        self.can_get_new = True

    def init_testcases(self):
        self.testcases = {
            task_type: [i for i in range(constants.NUM_DATA_POINTS)]
            for task_type in TaskType
        }
        # shuffle testcases
        if self.shuffle:
            for task_type in TaskType:
                random.shuffle(self.testcases[task_type])

    # add new items from fixed set of items (guaranteed order? random order?)
    def add_tasks(self, task_count: int):
        added_count = 0
        while added_count < task_count:
            try:
                task_type = random.choice(list(TaskType))
                self.queue.append(
                    {"type": task_type, "index": self.testcases[task_type].pop(0)}
                )
                added_count += 1
            except IndexError:
                # presumably we ran out of stuff, oh no, whatever
                print(f"we ran out of {task_type}")

    def get_task_data(self):
        if len(self.queue) == 0 or not self.can_get_new:
            return
        # the current task is the first in the queue
        first = self.queue[0]
        task_dir = self.data_dir / first["type"]
        with open(task_dir / first["type"].get_filename(first["index"]), "rb") as f:
            data = f.read()
        return {
            "type": "task",
            "task": first["type"],
            "b64": base64.b64encode(data).decode("ascii"),
        }

    def eval_task_result(
        self, data: dict[str, str | list[list[int]]], elapsed: float
    ) -> float:
        if len(self.queue) == 0:
            raise Exception("how did we eval_task_result with an empty queue?")
        prediction = data["result"]
        # the current task is the first in the queue
        first = self.queue.popleft()
        assert (
            first["type"] == data["task"]
        ), "The wrong type of task was returned, what happened?"
        task_dir = self.data_dir / first["type"]

        match first["type"]:
            case TaskType.ASR:
                with open(
                    task_dir / first["type"].get_gt_path(first["index"]), "r"
                ) as f:
                    contents = f.read()
                word_output = jiwer.process_words(
                    contents,
                    prediction,
                    reference_transform=wer_transforms,
                    hypothesis_transform=wer_transforms,
                )
                out = 1 - word_output.wer

            case TaskType.CV:
                if not prediction:
                    return 0

                for pred in prediction:
                    pred["image_id"] = first["index"]
                    pred["score"] = 1

                this_image_anns = self.get_cv_annotation(first["index"])
                ground_truth = COCOPatched(this_image_anns)
                results = ground_truth.loadRes(prediction)
                coco_eval = COCOeval(ground_truth, results, "bbox")
                coco_eval.evaluate()
                coco_eval.accumulate()
                coco_eval.summarize()
                out = coco_eval.stats[0].item()  # mAP@.5:.05:.95

            case TaskType.OCR:
                with open(
                    task_dir / first["type"].get_gt_path(first["index"]), "r"
                ) as f:
                    contents = f.read()
                cer = jiwer.cer(
                    contents,
                    prediction,
                    reference_transform=cer_transforms,
                    hypothesis_transform=cer_transforms,
                )
                out = 1 - cer

        # Use elapsed time to evaluate score
        speed_score = (
            max(constants.MAX_TIME_PER_TEST_CASE - elapsed, 0)
            / constants.MAX_TIME_PER_TEST_CASE
        )
        out = out * constants.PERFORMANCE_WEIGHT + speed_score * constants.SPEED_WEIGHT
        self.can_get_new = True
        return out


# upon completion of item, eval results
# send eval to frontend
# pop next item and do that one
# write results to file
