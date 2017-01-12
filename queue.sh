# Inception V3

# Inception V4

# Inception Resnet V2

# Resnet

if [ $1 -eq 3 ]
  then
    # GPU 3
    
    python "InceptionResnetV2 - Cat vs Dogs.py" --gpu 3 --name cvd_inception_resnet_v2_depth_150 --depth 150 --test --on-training
    # python "InceptionV3 - Cat vs Dogs.py" --gpu 3 --name cvd_inception_v3_depth_150_50 --depth 150 50 --test --on-training
    # python "InceptionV3 - Cat vs Dogs.py" --gpu 3 --name cvd_inception_v3_depth_200 --depth 200 --test --on-training
    # python "InceptionResnetV2 - Cat vs Dogs.py" --gpu 3 --name cvd_model_inception_resnet_v2_fulldata_2 --test --on-training
    # python "InceptionResnetV2 - Cat vs Dogs.py" --gpu 3 --name cvd_model_inception_resnet_v2_fulldata  --test --on-training
    # python "InceptionV4 - Cat vs Dogs - Deep FC.py" --gpu 3 --name cvd_model_inception_1 --test --on-training
    # python "InceptionV4 - Cat vs Dogs - Deep FC.py" --gpu 3 --name cvd_model_inception_deep_fc_1 --depth 100 --test --on-training
    # python "InceptionV4 - Cat vs Dogs - Deep FC.py" --gpu 3 --name cvd_model_inception_deep_fc_fulldata_1.Epoch.0 --depth 100 --test --on-training
    # python "InceptionV3 - Cat vs Dogs.py" --gpu 3 --name cvd_inception_v3_aux_logits --test --on-training --aux-logits 
    # python "InceptionV3 - Cat vs Dogs.py" --gpu 3 --name cvd_inception_v3_deep_logits_aux_logits --test --on-training --aux-logits --deep-logits
fi

if [ $1 -eq 2 ]
  then
    # GPU 2
    # python "InceptionV4 - Cat vs Dogs - Deep FC.py" --gpu 2 --name cvd_model_inception_deep_fc_fulldata_1.Epoch.1 --depth 128 --test --on-training
    # python "InceptionV4 - Cat vs Dogs - Deep FC.py" --gpu 2 --name cvd_model_inception_deep_fc_fulldata_2.Epoch.0 --depth 128 --test --on-training
    # python "InceptionV4 - Cat vs Dogs - Deep FC.py" --gpu 2 --name cvd_model_inception_deep_fc_fulldata_2.Epoch.1 --depth 128 --test --on-training
    # python "InceptionV4 - Cat vs Dogs - Deep Logits.py" --gpu 2 --name cvd_model_inception_deep_logits_1 --depth 128 --test --on-training
    # python "InceptionV4 - Cat vs Dogs - Deep Logits.py" --gpu 2 --name cvd_model_inception_deep_logits_mul_1 --aux mul --depth 128 --test --on-training
    # python "InceptionV4 - Cat vs Dogs - Deep FC.py" --gpu 2 --name cvd_inception_v4_depth_150 --depth 150 --test --on-training
    # python "Resnet - Cat vs Dogs.py" --gpu 2 --name cvd_model_resnet_101_fulldata --version 101 --test --on-training
    # python "InceptionV3 - Cat vs Dogs.py" --gpu 2 --name cvd_inception_v3_deep_logits --test --on-training --deep-logits
fi

if [ $1 -eq 1 ]
  then
    # GPU 1
    # python "Resnet - Cat vs Dogs.py" --gpu 1 --name cvd_model_resnet_12 --version 101 --test --on-training
    # python "Resnet - Cat vs Dogs.py" --gpu 1 --name cvd_model_resnet_14 --version 101 --test --on-training
    # python "Resnet - Cat vs Dogs.py" --gpu 1 --name cvd_model_resnet_152_fulldata --version 152 --test --on-training
    # python "Resnet - Cat vs Dogs.py" --gpu 1 --name cvd_model_resnet_15 --version 101 --test --on-training
    # python "Resnet - Cat vs Dogs.py" --gpu 1 --name cvd_model_resnet_16 --version 101 --test --on-training
    # python "Resnet - Cat vs Dogs.py" --gpu 1 --name cvd_model_resnet_17 --version 101 --test --on-training
    # python "Resnet - Cat vs Dogs.py" --gpu 1 --name cvd_model_resnet_18 --version 101 --test --on-training
    # python "InceptionV3 - Cat vs Dogs.py" --gpu 1 --name cvd_model_inception_v3_fulldata_2 --test --on-training
fi

if [ $1 -eq 0 ]
  then
    # GPU 0
    # python "Resnet - Cat vs Dogs.py" --gpu 0 --name cvd_model_resnet_19 --version 101 --test --on-training
    # python "Resnet - Cat vs Dogs.py" --gpu 0 --name cvd_model_resnet_20 --version 101 --test --on-training
    # python "Resnet - Cat vs Dogs.py" --gpu 0 --name cvd_model_resnet_21 --version 101 --test --on-training
    # python "Resnet - Cat vs Dogs.py" --gpu 0 --name cvd_model_resnet_bag_0 --version 101 --test --on-training
    # python "Resnet - Cat vs Dogs.py" --gpu 0 --name cvd_model_resnet_bag_1 --version 101 --test --on-training
    # python "Resnet - Cat vs Dogs.py" --gpu 0 --name cvd_model_resnet_bag_2 --version 101 --test --on-training
    # python "Resnet - Cat vs Dogs.py" --gpu 0 --name cvd_model_resnet_bag_3 --version 101 --test --on-training
    # python "InceptionV3 - Cat vs Dogs.py" --gpu 0 --name cvd_model_inception_v3_fulldata --test --on-training
fi
