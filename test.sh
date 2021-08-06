checkpoint=/data2/liukunhao/checkpoints/Adain/lr5e-5_bs8_s10_c1/iter159999loss_c1.3loss_s2.3.pth

CUDA_VISIBLE_DEVICES=$1 python3 test.py \
--content /data2/liukunhao/AdaIN-pytorch/content/lenna_cropped.jpg \
--style /data2/liukunhao/AdaIN-pytorch/style/en_campo_gris_cropped.jpg \
--checkpoint $checkpoint

CUDA_VISIBLE_DEVICES=$1 python3 test.py \
--content /data2/liukunhao/AdaIN-pytorch/content/avril_cropped.jpg \
--style /data2/liukunhao/AdaIN-pytorch/style/impronte_d_artista_cropped.jpg \
--checkpoint $checkpoint

CUDA_VISIBLE_DEVICES=$1 python3 test.py \
--content /data2/liukunhao/AdaIN-pytorch/content/chicago_cropped.jpg \
--style /data2/liukunhao/AdaIN-pytorch/style/ashville_cropped.jpg \
--checkpoint $checkpoint

CUDA_VISIBLE_DEVICES=$1 python3 test.py \
--content /data2/liukunhao/AdaIN-pytorch/content/cornell_cropped.jpg \
--style /data2/liukunhao/AdaIN-pytorch/style/woman_with_hat_matisse_cropped.jpg \
--checkpoint $checkpoint

CUDA_VISIBLE_DEVICES=$1 python3 test.py \
--content /data2/liukunhao/AdaIN-pytorch/content/modern_cropped.jpg \
--style /data2/liukunhao/AdaIN-pytorch/style/goeritz_cropped.jpg \
--checkpoint $checkpoint

CUDA_VISIBLE_DEVICES=$1 python3 test.py \
--content /data2/liukunhao/AdaIN-pytorch/content/sailboat_cropped.jpg \
--style /data2/liukunhao/AdaIN-pytorch/style/sketch_cropped.png \
--checkpoint $checkpoint