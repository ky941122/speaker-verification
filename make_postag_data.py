#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date   : 2019-07-30
# @Author : KangYu
# @File   : make_postag_data.py


import jieba.posseg as pseg


def make():

    f1 = open("/share/godeye/basic_data/course_text/course.txt", 'r')
    sents = f1.readlines()
    sents = [sent.strip() for sent in sents if sent.strip()]
    f1.close()

    print("read done.")

    f2 = open("/workspace/speaker_verification/dahai_postag_seg_2", 'w')
    i = 1
    for text in sents:
        tags = []
        words = pseg.cut(text)
        for w in words:
            tags.append(w.flag)
        tags = " ".join(tags)
        tags = tags.strip()

        f2.write(tags + "\n")
        print(i)
        i += 1

    f2.close()


if __name__ == "__main__":
    make()
