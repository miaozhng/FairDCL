# FairDCL


Fair dense representation in contrastive learning (FairDCL) is a fairness optimization approach build on MoCo-v2 contrastive learning framework. The core codes are building sensitive attribute embeddings and deriving discriminator loss.

###
Train:
```
python3 main_moco_fair.py \
-a resnet50 \
--lr 0.001 \
--batch-size 32 \
--moco-dim 512 \
--sensitive_id1 "" \
--sensitive_id2 "" \
--mlp --moco-t 0.2 --aug-plus --cos \
  "Data path for pre-training"
```

````--sensitive_id1````  (Numpy data path for the first sensitive attribute including image ids)

````--sensitive_id2````  (Numpy data path for the second sensitive attribute including image ids)


TODO:
1) Making the number of sensitive attribute classes adjustable.
2) Providing codes for comparison methods: Gradient reversal and Domain independent training.
3) Downstream fine-tuning codes.
