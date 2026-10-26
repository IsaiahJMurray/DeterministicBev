# Emptied by patches/bevformer/09_dd3d_init_empty.sh
# Reason: dd3d/modeling は Detectron2 依存。BEVFormer 推論パスでは dd3d 不要、
#         かつ v5 image は Detectron2 未 install。plugin load 時の ImportError 回避。
