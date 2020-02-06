
for i in {2..100};
do python3 config_generator.py 200 10 5 train/config${i}.txt;
done;

