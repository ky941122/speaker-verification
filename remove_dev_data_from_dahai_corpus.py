#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date   : 2019-08-01
# @Author : KangYu
# @File   : remove_dev_data_from_dahai_corpus.py

import pickle

if __name__ == "__main__":

    f1 = open("/workspace/speaker_verification/data/dahai/test/voice_attention_tag.pkl", 'rb')
    dev_dh = pickle.load(f1)
    f1.close()

    f2 = open("/share/godeye/basic_data/course_text/student.txt", 'r')
    students = f2.readlines()
    f2.close()
    students = [sent.strip() for sent in students if sent.strip()]

    f3 = open("/share/godeye/basic_data/course_text/teacher.txt", 'r')
    teachers = f3.readlines()
    f3.close()
    teachers = [sent.strip() for sent in teachers if sent.strip()]

    i = 1
    for course_content in dev_dh:
        for sent in course_content:
            _, label, text = sent
            text = text.strip()

            if label == 0:
                while text in students:
                    students.remove(text)
            elif label == 1:
                while text in teachers:
                    teachers.remove(text)
            else:
                while text in teachers:
                    teachers.remove(text)
                while text in students:
                    students.remove(text)

            print(i)
            i += 1


    f22 = open("/workspace/speaker_verification/data/student_without_dev", 'w')
    for sent in students:
        f22.write(sent + "\n")
    f22.close()

    f33 = open("/workspace/speaker_verification/data/teacher_without_dev", 'w')
    for sent in teachers:
        f33.write(sent + "\n")
    f33.close()

