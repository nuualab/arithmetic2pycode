# -*- coding: utf-8 -*-
# + active=""
# import re
# -

number_dict = {
    0: '①',
    1: '②',
    2: '③',
    3: '④',
    4: '⑤',
    5: '⑥',
    6: '⑦',
    7: '⑧',
    8: '⑨'
}

number_list = ['①', '②', '③', '④', '⑤', '⑥', '⑦', '⑧', '⑨']


def get_number_character(index):
    if index in number_dict:
        return number_dict[index]
    else:
        return None


def extract_numbers(text):
    regex_number = re.compile('[0-9]{1,1000}.[0-9]{1,1000}|'
                              '[0-9]{1,1000}/[0-9]{1,1000}|'
                              '[0-9]{1,1000}')

    n_idx = 0
    n_dict = dict()
    while re.search(regex_number, text) is not None:
        search = re.search(regex_number, text)
        prev_c = ''
        next_c = ''
        if search.start() > 0:
            prev_c = text[search.start() - 1]
        if search.end() < len(text):
            next_c = text[search.end()]

        dst = get_number_character(n_idx)

        if prev_c != ' ' and prev_c != '':
            dst = ' ' + dst
        if next_c != ' ' and next_c != '':
            dst = dst + ' '

        text = text[0:search.start()] + dst + text[search.end():]

        n_idx += 1
        n_dict[dst.replace(' ', '')] = search.group() if '/' not in search.group() else '(' + search.group() + ')'

    return text, n_dict
