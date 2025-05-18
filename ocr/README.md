# OCR

Your OCR challenge is to read text in a scanned document.

This Readme provides a brief overview of the interface format; see the Wiki for the full [challenge specifications](https://github.com/til-ai/til-25/wiki/Challenge-specifications).

## Our Solution

### Qualifiers

- OCR pipeline from [docTR](https://github.com/mindee/doctr) with fine-tuning on the provided train set, using the lightest models (pipeline only has <70M parameters)to maximise speed
- Image preprocessing to get rid of the bleedthrough (did not show significant improvement on ablations when used in conjunction with the fine-tuned detection model)
- Spellchecker with levenshtein edit distance to correct out-of-vocab predictions
- "Shortcut" by only processing the top section of the image since we noticed there were only 5 possible texts in the train set, falling back to full inference if a match isn't found

Detection Model: [LinkNet with a ResNet18 Backbone](https://mindee.github.io/doctr/modules/models.html#doctr.models.detection.linknet_resnet18)
Recognition Model: [CRNN with a MobileNet V3 Small Backbone](https://mindee.github.io/doctr/modules/models.html#doctr.models.recognition.crnn_mobilenet_v3_small)

### Fine-Tuning Parameters

Fine-tuned using docTR's training scripts for [detection](https://github.com/mindee/doctr/blob/main/references/detection/train_pytorch.py) and [recognition](https://github.com/mindee/doctr/blob/main/references/recognition/train_pytorch.py)

#### Detection

- Epochs: 3
- Batch size: 8
- Learning rate: 1e-4 with cosine decay
- Optimizer: AdamW
- With AMP @ float16

#### Recognition

- Epochs: 1
- Batch size: 1024
- Learning rate: 1e-4 with cosine decay
- Optimizer: AdamW
- With AMP @ float16

## Input

The input is sent via a POST request to the `/ocr` route on port 5003. It is a JSON document structured as such:

```JSON
{
  "instances": [
    {
      "key": 0,
      "b64": "BASE64_ENCODED_IMAGE"
    },
    ...
  ]
}
```

The `b64` key of each object in the `instances` list contains the base64-encoded bytes of the input image in JPEG format. The length of the `instances` list is variable.

## Output

Your route handler function must return a `dict` with this structure:

```Python
{
    "predictions": [
        "Predicted transcript one.",
        "Predicted transcript two.",
        ...
    ]
}
```

where each string in `predictions` is the predicted OCR transcription for the corresponding audio file.

The $k$-th element of `predictions` must be the prediction corresponding to the $k$-th element of `instances` for all $1 \le k \le n$, where n is the number of input instances. The length of `predictions` must equal that of `instances`.