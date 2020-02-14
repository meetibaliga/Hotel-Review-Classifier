#making sets
for i in {1..25}
do
python3 baliga-meeti-assgn2.py make_sets
python3 baliga-meeti-assgn2.py train
cat final_weights.csv
python3 baliga-meeti-assgn2.py test
done
