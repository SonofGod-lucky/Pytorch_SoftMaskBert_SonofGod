#coding=utf-8
from pretrain_config import *
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET


def load_data(input_file):
    contents = []
    passages = []
    reader = open(input_file)
    text = ''
    line = reader.readline()
    while line:
        line = line.strip()
        if line.startswith("</SENTENCE>"):
            passages.append(line)
            sentence = ET.fromstringlist(passages)
            if not text:
                passages.clear()
                continue
            # text = sentence.findtext("TEXT")
            content = {"text": text, "mistakes": []}
            for mistake in sentence.iter("MISTAKE"):
                wrong = mistake.findtext("WRONG")
                correct = mistake.findtext("CORRECTION")
                if wrong == correct:
                    continue
                reform = {"wrong": wrong, "correct": correct, "loc": mistake.findtext("LOCATION")}
                content["mistakes"].append(reform)
            if len(content["mistakes"]) > 0:
                contents.append(content)
            passages = []
            text = ''
        elif line.startswith("<TEXT>"):
            text = line[len('<TEXT>'):-len('</TEXT>')]
        elif line:
            passages.append(line)
        line = reader.readline()
    reader.close()
    return contents

def writer(path,datas):
    with open(path,'w',encoding='utf-8') as writer:
        for data in datas:
            text = data['text']
            chars = list(text)
            for mistake in data['mistakes']:
                index = int(mistake['loc']) -1
                chars[index] = mistake['correct']
            correct = ''.join(chars)
            mistake_labels = ''.join(['0' if str(text[i]) == str(correct[i]) else '1' for i in range(len(text))])
            line = text + '-***-' + correct + '-***-' + mistake_labels
            writer.write(line+'\n')


if __name__ == '__main__':
    datas = load_data(path_train_data)
    print(len(datas))
    writer(path_train_save,datas[:200])
