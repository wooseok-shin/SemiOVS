# scrips / num gpu / port / splits


# Training on Pascal VOC
sh scripts/train_pascal.sh 1 11 92
wait

# Training on Pascal Context
sh scripts/train_pc60.sh 1 11 1_32
wait