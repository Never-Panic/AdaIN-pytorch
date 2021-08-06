CUDA_VISIBLE_DEVICES=$1 python3 test.py \
--content /data2/liukunhao/AdaIN-pytorch/content/modern_cropped.jpg \
--style /data2/liukunhao/AdaIN-pytorch/style/ashville_cropped.jpg \
--checkpoint /data2/liukunhao/checkpoints/Adain/lr5e-5_bs8_s10_c1/iter109999loss_c1.3loss_s2.9.pth