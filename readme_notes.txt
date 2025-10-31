Get Started:
https://yolox.readthedocs.io/en/latest/quick_run.html#reproduce-our-results-on-coco

Python 3.9 (or 3.8) recommended, 10 fails: https://github.com/Megvii-BaseDetection/YOLOX/issues/1585
https://github.com/Megvii-BaseDetection/YOLOX/issues/87#issuecomment-885024757
Done: Python 3.9.24

conda create -n yolox python=3.9
conda activate yolox
python -m pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 --index-url https://download.pytorch.org/whl/cu111
# python -m pip install -r requirements.txt
python -m pip install --no-deps -r requirements.txt
pip uninstall -y yolox || true
rm -rf build *.egg-info
# python -m pip install -v -e .
pip install -e . --no-build-isolation --config-settings editable_mode=compat
# --no-build-isolation lets YOLOX’s build see your installed torch, avoiding the earlier “torch required” and “ModuleNotFoundError: yolox” hiccups.
python -c "import torch, torchvision, yolox; print(torch.__version__, torchvision.__version__, 'YOLOX OK')"
python -m pip install cython
python -m pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
python -m pip install wandb
wandb login
<login to your W&B account>

Test:
python tools/demo.py image -n yolox-s -c /path/to/your/yolox_s.pth --path assets/dog.jpg --conf 0.25 --nms 0.45 --tsize 640 --save_result --device [cpu/gpu]
python tools/demo.py image -n yolox-nano -c /home/ramesh/automltraining/yolox/pytorch_weights/yolox_nano.pth --path assets/dog.jpg --conf 0.25 --nms 0.45 --tsize 416 --save_result --device gpu