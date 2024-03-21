CUDA_VISIBLE_DEVICES=0 python3 -m scripts.train_field field=grid dataset=image field.side_length=33
# CUDA_VISIBLE_DEVICES=0 python3 -m scripts.train_field field=siren dataset=image learning_rate=1e-4
# CUDA_VISIBLE_DEVICES=0 python3 -m scripts.compare_fields