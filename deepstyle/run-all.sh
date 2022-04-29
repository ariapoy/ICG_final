# run all experiments
# h * w = 182040 for 8 GB memory
# h * w = 273060 for 12 GB memory

# at luhome
# bash run.sh "Lift_Poy-Lu_resized" 370 493 # 454 605
bash run.sh "101" 339 452; # 339 452
bash run.sh "101_night" 349 522; # 427 640
bash run.sh "building" 357 510; # 561 800
bash run.sh "flower" 369 493; # 450 600

# at CLLab
# bash run.sh "Life_YaXu-Liu" 425 425; # 532 532
# bash run.sh "Life_Yui-Aragaki" 560 331; # 1120 663
# bash run.sh "NTU" 512 361 # 1024 723
# bash run.sh "NTU_night" 492 370 # 720 540
# bash run.sh "Avatar_Jimmy-Yang" 375 375; # 750 750
