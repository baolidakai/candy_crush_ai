
for i in {2..100};
do python3 config_generator.py 200 10 5 eval/config${i}.txt;
done;

