import re

number_dict = {
    0: '①',
    1: '②',
    2: '③',
    3: '④',
    4: '⑤',
    5: '⑥',
    6: '⑦',
    7: '⑧',
    8: '⑨',
    9: 'ⓐ',
    10: 'ⓑ',
    11: 'ⓒ'
}

number_list = ['①', '②', '③', '④', '⑤', '⑥', '⑦', '⑧', '⑨', 'ⓐ', 'ⓑ', 'ⓒ']

unit_dict = {
    'kcal': '킬로칼로리',
    'cal': '칼로리',
    'mm2': '제곱밀리미터',
    'cm2': '제곱센티미터',
    'km2': '제곱킬로미터',
    'mm3': '세제곱밀리미터',
    'cm3': '세제곱센티미터',
    'km3': '세제곱킬로미터',
    'mm': '밀리미터',
    'cm': '센티미터',
    'km': '킬로미터',
    'm2': '제곱미터',
    'm3': '세제곱미터',
    'lb': '파운드',
    'ml': '밀리리터',
    'mg': '밀리그램',
    'kg': '킬로그램',
    '㎣': '세제곱밀리미터',
    '㎤': '세제곱센티미터',
    '㎥': '세제곱미터',
    '㎦': '세제곱킬로미터',
    '㎟': '제곱밀리미터',
    '㎠': '제곱센티미터',
    '㎡': '제곱미터',
    '㎢': '제곱킬로미터',
    '㎜': '밀리미터',
    '㎝': '센티미터',
    '㎞': '킬로미터',
    '㎎': '밀리그램',
    '㎏': '킬로그램',
    '㎈': '칼로리',
    '㎉': '킬로칼로리',
    'g': '그램',
    'm': '미터',
    'l': '리터',
    '%': '퍼센트',
}

regex_number = re.compile('[0-9]{1,100}\.[0-9]{1,100}|'
                          '[0-9]{1,100}/[0-9]{1,100}|'
                          '[0-9]{1,100}')
regex_number_replaced = re.compile('①|②|③|④|⑤|⑥|⑦|⑧|⑨|ⓐ|ⓑ|ⓒ')
# regex_equation = re.compile('[A-Z0-9]{1,100}[ ]{0,1}[+\-*\/=].*[+\-*\/=][ ]{0,1}[A-Z0-9]{1,100}|'
#                             '[A-Z0-9]{1,100}[ ]{0,1}[<>=]{1,2}[ ]{0,1}[A-Z0-9]{1,100}')
regex_unknown_num = re.compile('[0-9A-Z]{2,100}')
regex_unit = re.compile('[kK][cC][aA][lL]|[cC][aA][lL]|'
                        '[mM][mM]2|[cC][mM]2|[kK][mM]2|'
                        '[mM][mM]3|[cC][mM]3|[kK][mM]3|'
                        '[mM][mM]|[cC][mM]|[kK][mM]|'
                        '[mM]2|[mM]3|[lL][bB]|[mM][lL]|'
                        '[mM][gG]|[kK][gG]|'
                        '㎣|㎤|㎥|㎦|㎟|㎠|㎡|㎢|㎜|㎝|㎞|㎎|㎏|㎈|㎉'
                        '[gG]|[mM]|[lL]|%')
regex_unknown_variable = re.compile('[A-Z]')
regex_month = re.compile('[0-9]{1,2}월')


def get_number_character(index):
    if index in number_dict:
        return number_dict[index]
    else:
        return None


def preprocess(text, n_dict):

    # 단위 치환
    text = convert_unit_expression(text)

    # 수식 편집
    text = convert_equation(text)

    # 숫자 치환
    # 학습데이터라면, 이미 매핑되어 있는 number_dict 를 인자로 넣어주고, 그렇지 않다면 빈 dict 를 추가한다.
    extraction = extract_numbers(text, n_dict)
    text = extraction[0]
    n_dict = extraction[1]

    # 대문자 A, B .. 띄우기
    text = trim_unknown_variable(text)

    # 마지막 공백 trim 처리
    text = text.replace('  ', ' ').strip()

    # 연산기호 치환
    text = text.replace('✕', '*').replace('×', '*').replace('÷', '/')

    # quotation 치환
    text = text.replace('"', '\'')
    return text, n_dict


def convert_unit_expression(text):

    while re.search(regex_unit, text) is not None:
        search = re.search(regex_unit, text)
        unit = search.group()
        dst = unit_dict[unit.lower()]

        text = text[0:search.start()] + dst + text[search.end():]

    return text


def convert_equation(text):
    c_idx = 0
    while re.search(regex_unknown_num, text[c_idx:]) is not None:
        search = re.search(regex_unknown_num, text[c_idx:])

        s_idx = c_idx + search.start()
        e_idx = c_idx + search.end()

        if len(re.findall('[A-Z]', search.group())) == 0:
            c_idx = e_idx
            continue

        dst = ' '.join(list(search.group()))
        text = text[0:s_idx] + dst + text[e_idx:]
        c_idx = s_idx + len(dst)

    return text


def extract_numbers(text, n_dict):

    rn_list = set(re.findall(regex_number_replaced, text))
    record = list()
    c_idx = 0   # cursor_index
    n_idx = len(rn_list)   # number_index

    m_count = len(set(re.findall(regex_month, text)))

    # 기존에 치환된 숫자 중 중복된 숫자들을 개별 숫자로 치환한다. n_dict 에 해당 매핑을 추가한다.
    while re.search(regex_number_replaced, text[c_idx:]) is not None:
        search = re.search(regex_number_replaced, text[c_idx:])

        s_idx = c_idx + search.start()
        e_idx = c_idx + search.end()

        if search.group() not in record:
            record.append(search.group())
            c_idx = e_idx
            continue

        dst = get_number_character(n_idx)
        text = text[0:s_idx] + dst + text[e_idx:]
        c_idx = s_idx + 1
        n_dict[dst] = n_dict[search.group()]
        n_idx += 1

    c_idx = 0   # cursor_index
    # 치환되지 않은 숫자들을 모두 뽑아 치환하고 매핑한다.
    while re.search(regex_number, text[c_idx:]) is not None:
        search = re.search(regex_number, text[c_idx:])

        s_idx = c_idx + search.start()
        e_idx = c_idx + search.end()

        # 1 의 경우 매핑하지 않는다.
        if search.group() == '1':
            c_idx = e_idx
            continue

        # 0~9 를 나타내는 경우, 0과 9를 매핑하지 않는다.
        if search.group() == '0':
            n_search = re.search(regex_number, text[e_idx:])
            if n_search is not None and n_search.group() == '9':
                bet_str = text[search.end():n_search.start()].replace(' ', '')
                if bet_str == '에서' or bet_str == '부터' or bet_str == '으로부터':
                    c_idx = e_idx + n_search.end()
                    continue

        prev_c = ''
        next_c = ''
        if search.start() > 0:
            prev_c = text[s_idx - 1]
        if search.end() < len(text):
            next_c = text[e_idx]

        dst = get_number_character(n_idx)

        if prev_c != ' ' and prev_c != '':
            dst = ' ' + dst
        if next_c != ' ' and next_c != '':
            dst = dst + ' '

        # 7월.. 8월.. 의 경우 월이 하나만 등장하면 치환하지 않는다.
        if m_count == 1 and next_c == '월':
            c_idx = e_idx
            continue

        text = text[0:s_idx] + dst + text[e_idx:]
        c_idx = s_idx + 1

        n_idx += 1
        n_dict[dst.replace(' ', '')] = search.group() if '/' not in search.group() else '(' + search.group() + ')'

    return text, n_dict


def trim_unknown_variable(text):

    c_idx = 0   # cursor_index
    # 대문자 알파벳으로 된 미지수는 띄어쓰기 한다.
    while re.search(regex_unknown_variable, text[c_idx:]) is not None:
        search = re.search(regex_unknown_variable, text[c_idx:])

        s_idx = c_idx + search.start()
        e_idx = c_idx + search.end()

        prev_c = ''
        next_c = ''
        if search.start() > 0:
            prev_c = text[s_idx - 1]
        if search.end() < len(text):
            next_c = text[e_idx]

        dst = search.group()

        if prev_c != ' ' and prev_c != '':
            dst = ' ' + dst
        if next_c != ' ' and next_c != '':
            dst = dst + ' '

        text = text[0:s_idx] + dst + text[e_idx:]
        c_idx = s_idx + 1

    return text





