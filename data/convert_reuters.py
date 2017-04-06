import os

src_folder = './reuters'
out_folder = './reuters_csv'

os.makedirs(out_folder, exist_ok=True)

fcats = open(src_folder + '/cats.txt', 'rt')
ftrain = open(out_folder + '/train.csv', 'wt')
ftest = open(out_folder + '/test.csv', 'wt')
fclass = open(out_folder + '/classes.txt', 'wt')
topic_index = {}
clines = fcats.readlines()

for line in clines:
    print(line[:-1])
    src_file = src_folder + '/' + line.split(' ')[0]
    fsrc = open(src_file, 'rt')
    try:
        src = fsrc.readlines()
    except UnicodeDecodeError:
        fsrc.close()
        continue

    title = src[0].strip()
    content = [s.strip() for s in src[1:]]
    content = ' '.join(content)

    topic = line.split(' ')[1].strip()
    if topic not in topic_index:
        topic_index[topic] = len(topic_index)

    s = '"%s","%s","%s"\n' % (topic_index[topic]+1, title, content)
    if line[:4] == 'test':
        ftest.write(s)
    else:
        ftrain.write(s)
    fsrc.close()

topics = [''] * len(topic_index)
for t in topic_index.keys():
    topics[topic_index[t]] = t
fclass.write('\n'.join(topics))

fcats.close()
ftrain.close()
ftest.close()
fclass.close()
