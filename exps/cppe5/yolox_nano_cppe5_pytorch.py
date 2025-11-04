# exps/example/custom/yolox_nano_cppe5.py
import os
import torch.nn as nn
from yolox.exp import Exp as MyExp
from yolox.data import get_yolox_datadir

class Exp(MyExp):
    def __init__(self):
        super().__init__()

        # ------- model size: NANO defaults -------
        self.depth = 0.33
        self.width = 0.25
        self.input_size = (416, 416)
        self.test_size = (416, 416)
        self.random_size = (10, 20)
        self.mosaic_scale = (0.5, 1.5)
        self.mosaic_prob = 0.5
        self.enable_mixup = False

        # ------- custom dataset -------
        self.num_classes = 5 # CPPE-5 => 5 classes
        # symlink dataset to YOLOX/datasets/cppe5, to resolve automatically:
        # ln -s /home/ramesh/tfds_datasets/cppe5 /home/ramesh/automltraining/yolox/YOLOX/datasets/cppe5
        self.data_dir = os.path.join(get_yolox_datadir(), "cppe5")
        self.train_ann = "coco_train.json"
        self.val_ann   = "coco_validation.json"
        self.test_ann  = "coco_test.json"

        # book-keeping
        self.exp_name = os.path.splitext(os.path.basename(__file__))[0]
        # (Optional training knobs)
        self.max_epoch = 300  # 300
        self.no_aug_epochs = 1  # 15
        self.warmup_epochs = 3  # 5

    # NANO uses depthwise conv in backbone+head
    def get_model(self, sublinear=False):
        from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if getattr(self, "model", None) is None:
            in_channels = [256, 512, 1024]
            backbone = YOLOPAFPN(
                self.depth, self.width, in_channels=in_channels, act=self.act, depthwise=True
            )
            head = YOLOXHead(
                self.num_classes, self.width, in_channels=in_channels, act=self.act, depthwise=True
            )
            self.model = YOLOX(backbone, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        return self.model

    # TRAIN split
    def get_dataset(self, cache: bool, cache_type: str = "ram"):
        from yolox.data import COCODataset, TrainTransform
        return COCODataset(
            data_dir=self.data_dir,          # cppe5/
            json_file=self.train_ann,        # "coco_train.json"
            name="images",                   # <== important name="images" ==> YOLOX’s COCO loader joins data_dir / name / file_name_from_json
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=120,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob,
            ),
            cache=cache,
            cache_type=cache_type,
        )

    # VAL split
    def get_eval_dataset(self, **kwargs):
        from yolox.data import COCODataset, ValTransform
        legacy = kwargs.get("legacy", False)
        return COCODataset(
            data_dir=self.data_dir,
            json_file=self.val_ann,          # "coco_validation.json"
            name="images",
            img_size=self.test_size,
            preproc=ValTransform(legacy=legacy),
        )

    # TEST split (optional)
    def get_test_dataset(self, **kwargs):
        from yolox.data import COCODataset, ValTransform
        legacy = kwargs.get("legacy", False)
        return COCODataset(
            data_dir=self.data_dir,
            json_file=self.test_ann,         # "coco_test.json"
            name="images",
            img_size=self.test_size,
            preproc=ValTransform(legacy=legacy),
        )


# To use this exp file, for example:
# (optional) login to W&B once
# wandb login
# Train:

# rename the previous output folder before running this, as it overrides existing folder
# cd /home/ramesh/automltraining/yolox/YOLOX
# python tools/train.py -f exps/cppe5/yolox_nano_cppe5_pytorch.py -d 1 -b 64 --fp16 -o --cache --logger wandb wandb-project yolox-cppe5 -c weights/cppe5_yolox_nano.pth

# # evaluate on validation split
# python tools/eval.py -f exps/cppe5/yolox_nano_cppe5_pytorch.py -c YOLOX_outputs/yolox_nano_cppe5_pytorch/latest_ckpt.pth -d 1 --conf 0.01 --nms 0.65

# # evaluate on test split
# python tools/eval.py -f exps/cppe5/yolox_nano_cppe5_pytorch.py -c YOLOX_outputs/yolox_nano_cppe5_pytorch/latest_ckpt.pth -d 1 --conf 0.01 --nms 0.65 --test

# # export (optional)
# python tools/export_onnx.py \
#   -f exps/cppe5/yolox_nano_cppe5_pytorch.py \
#   -c YOLOX_outputs/yolox_nano_cppe5_pytorch/best_ckpt.pth \
#   --output-name yolox_nano_cppe5_pytorch.onnx --dynamic



# /home/ramesh/automltraining/yolox/YOLOX/YOLOX_outputs_pytorch_cppe5/yolox_nano_cppe5_pytorch/train_log.txt