CUDA_VISIBLE_DEVICES=0 python3 -m scripts.train_field field=ground_plan \
                                    dataset=implicit \
                                    dataset.function=torus \
                                    field.d_grid_feature=128 \
                                    field.positional_encoding_octaves=8 \
                                    field.grid.side_length=65 \
                                    field.mlp.positional_encoding_octaves=null \
                                    field.mlp.num_hidden_layers=2 \
                                    field.mlp.d_hidden=64 \

# CUDA_VISIBLE_DEVICES=0 python3 -m scripts.train_field field=hybrid_grid dataset=image field.d_grid_feature=128 field.grid.side_length=33 field.mlp.positional_encoding_octaves=null field.mlp.num_hidden_layers=2 field.mlp.d_hidden=64
# CUDA_VISIBLE_DEVICES=0 python3 -m scripts.train_field field=grid dataset=image field.side_length=33
# CUDA_VISIBLE_DEVICES=0 python3 -m scripts.train_field field=siren dataset=image learning_rate=1e-4
# CUDA_VISIBLE_DEVICES=0 python3 -m scripts.compare_fields