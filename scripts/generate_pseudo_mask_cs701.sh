# NEED TO SET
DATASET_ROOT=/home/hoangle/cs701-wsss/public
GPU=0

# # # 1. train a classification network and compute refined seed
CUDA_VISIBLE_DEVICES=${GPU} python3 run_sample_cs701.py \
    --dataset_root ${DATASET_ROOT} \
    --num_workers 8 \
    --cam_eval_thres 0.15 \
    --cam_to_ir_label_pass True \
    --make_cam_pass True \
    --conf_fg_thres 0.30 \
    --conf_bg_thres 0.05 
    # --train_cam_pass True \
#     # --eval_cam_pass True \


# # 1.1 infer labels 
CUDA_VISIBLE_DEVICES=${GPU} python3 run_sample_cs701.py \
    --dataset_root ${DATASET_ROOT} \
    --infer_labels_pass True 


# # 2.1. train an attribute manipulation network
CUDA_VISIBLE_DEVICES=${GPU} python3 run_sample_cs701.py \
    --dataset_root ${DATASET_ROOT} \
    --num_workers 8 \
    --train_amn_pass True

# # 2.2. generate activation maps and refined seed for boundary refinement
# CUDA_VISIBLE_DEVICES=${GPU} python3 run_sample_cs701.py \
#     --dataset_root ${DATASET_ROOT} \
#     --num_workers 8 \
#     --make_amn_cam_pass True \
#     --eval_amn_cam_pass True \
#     --amn_cam_to_ir_label_pass True \
#     --conf_fg_thres 0.45 \
#     --conf_bg_thres 0.15

# # 3.1. train a boundary refinement network (IRN)
# CUDA_VISIBLE_DEVICES=${GPU} python3 run_sample_cs701.py \
#     --dataset_root ${DATASET_ROOT} \
#     --num_workers 8 \
#     --train_irn_pass True

# # 3.2. generate the final pseudo-masks
# CUDA_VISIBLE_DEVICES=${GPU} python3 run_sample_cs701.py \
#     --dataset_root ${DATASET_ROOT} \
#     --num_workers 8 \
#     --make_sem_seg_pass True \
#     --eval_sem_seg_pass True
