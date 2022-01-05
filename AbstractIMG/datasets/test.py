import os
# os.makedirs('./deviantArt_loader')
# os.makedirs('./deviantArt_loader/neg')
# os.makedirs('./deviantArt_loader/pos')
neg, pos = [], []
with open('./deviantArt/scores.txt', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            filename, label = line.strip().split('\t')
            label = int(label)
            if label == 0:
                neg.append(filename)
            else:
                pos.append(filename)

for i in neg:
    os.system(f'cp ./deviantArt/{i} ./deviantArt_loader/neg/{i}') 

for i in pos:
    os.system(f'cp ./deviantArt/{i} ./deviantArt_loader/pos/{i}') 
