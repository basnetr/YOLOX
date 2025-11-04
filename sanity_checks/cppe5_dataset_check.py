from yolox.data.datasets.coco import COCODataset
ds = COCODataset(
    data_dir="/home/ramesh/tfds_datasets/cppe5",   # or your absolute path
    json_file="coco_train.json",
    name="images",
    img_size=(640, 640),
    preproc=None
)
print(len(ds), ds.annotations[0][3])

ds = COCODataset(
    data_dir="/home/ramesh/tfds_datasets/cppe5",   # or your absolute path
    json_file="coco_validation.json",
    name="images",
    img_size=(640, 640),
    preproc=None
)
print(len(ds), ds.annotations[0][3])

ds = COCODataset(
    data_dir="/home/ramesh/tfds_datasets/cppe5",   # or your absolute path
    json_file="coco_test.json",
    name="images",
    img_size=(640, 640),
    preproc=None
)
print(len(ds), ds.annotations[0][3])