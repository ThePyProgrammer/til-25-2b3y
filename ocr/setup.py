from doctr.models import ocr_predictor
import layoutparser as lp


detection_arch = [
    "db_resnet34",
    "db_resnet50",
    "db_mobilenet_v3_large",
    "linknet_resnet18",
    "linknet_resnet34",
    "linknet_resnet50",
    "fast_tiny",
    "fast_small",
    "fast_base",
]

recognition_arch = [
    "crnn_vgg16_bn",
    "crnn_mobilenet_v3_small",
    "crnn_mobilenet_v3_large",
    "sar_resnet31",
    "master",
    "vitstr_small",
    "vitstr_base",
    "parseq",
]

if __name__ == "__main__":
    ocr_model = ocr_predictor(detection_arch[2], recognition_arch[0], pretrained=True)

    lp_model = lp.models.Detectron2LayoutModel(
                config_path ='lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config', # In model catalog
                label_map   = {0: "Text", 1: "Title"}, # In model`label_map`
                extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.7] # Optional
            )
