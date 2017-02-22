import os
import numpy as np

DIRECTORY = './bbc_csv'
SOURCE_DIR = './bbc'
if not os.path.exists(DIRECTORY):
    print('Creating %s' % DIRECTORY)
    os.mkdir(DIRECTORY)

topic_dirs = os.listdir(SOURCE_DIR)
topic_dirs.remove('README.TXT')
outputs = []
f_class = open(DIRECTORY + '/classes.txt', mode='wt')
f_csv = open(DIRECTORY + '/train.csv', mode='wt')

for idx, dir in enumerate(topic_dirs):
    f_class.write(dir.capitalize() + '\n')
    files = os.listdir(SOURCE_DIR + '/' + dir)

    for fn in files:
        f = open(SOURCE_DIR + '/' + dir + '/' + fn, 'rt')
        try:
            lines = f.readlines()
        except UnicodeDecodeError:
            f.close()
            continue
        f.close()
        title = lines[0].strip()
        content = ''
        for line in lines[1:]:
            content += line.strip() + ' '

        out = '"%s","%s","%s"\n' % (idx+1, title, content)
        outputs.append(out)

np.random.shuffle(outputs)

for out in outputs:
    print(out)
    f_csv.write(out)

f_class.close()
f_csv.close()
