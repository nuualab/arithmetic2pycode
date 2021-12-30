import re
import parso

iter_var_sequence = ['i', 'j', 'k', 'q', 'p', 'l']

regex_for_syntax = re.compile('for [a-z] in [^ ]{1,100}')
regex_for_merged_syntax = re.compile('f\$[0-9]{1,2}\$[^ ]{1,100}')
regex_not_dup_constraint = re.compile('[ijkqpl] != [ijkqpl]')
regex_not_zero_constraint = re.compile('[ijkqpl] != 0|'
                                       '0 != [ijkqpl]')

regex_encoded_multiple_iteration = re.compile('loop')
regex_encoded_not_dup_constraint = re.compile('uniq')
regex_encoded_not_zero_constraint = re.compile('nz')

regex_reserved_range = re.compile('range\(0,10\)|range\(10\)|range\(0,9\+1\)|range\(9\+1\)|'
                                  'range\(10,100\)|'
                                  'range\(100,1000\)|'
                                  'range\(1000,10000\)')
regex_reserved_range_10 = re.compile('range\(0,10\)|range\(10\)|range\(0,9\+1\)|range\(9\+1\)')
regex_reserved_range_100 = re.compile('range\(10,100\)')
regex_reserved_range_1000 = re.compile('range\(100,1000\)')
regex_reserved_range_10000 = re.compile('range\(1000,10000\)')
regex_reserved_range_list = [regex_reserved_range_10, regex_reserved_range_100, regex_reserved_range_1000, regex_reserved_range_10000]

regex_encoded_reserved_range = re.compile('r_0_10|r_10_100|r_100_1000|r_1000_10000')
regex_encoded_reserved_range_10 = re.compile('r_0_10')
regex_encoded_reserved_range_100 = re.compile('r_10_100')
regex_encoded_reserved_range_1000 = re.compile('r_100_1000')
regex_encoded_reserved_range_10000 = re.compile('r_1000_10000')
regex_encoded_reserved_range_list = [regex_encoded_reserved_range_10, regex_encoded_reserved_range_100, regex_encoded_reserved_range_1000, regex_encoded_reserved_range_10000]

regex_digit = re.compile('i \* 10000 \+ j \* 1000 \+ k \* 100 \+ q \* 10 \+ p|'
                           'i \* 1000 \+ j \* 100 \+ k \* 10 \+ q|'
                           'i \* 100 \+ j \* 10 \+ k|'
                           'i \* 10 \+ j')
regex_digit_5 = re.compile('i \* 10000 \+ j \* 1000 \+ k \* 100 \+ q \* 10 \+ p')
regex_digit_4 = re.compile('i \* 1000 \+ j \* 100 \+ k \* 10 \+ q')
regex_digit_3 = re.compile('i \* 100 \+ j \* 10 \+ k')
regex_digit_2 = re.compile('i \* 10 \+ j')
regex_digit_list = [regex_digit_2,
                    regex_digit_3,
                    regex_digit_4,
                    regex_digit_5]

regex_encoded_digit = re.compile('d_5|d_4|d_3|d_2')
regex_encoded_digit_5 = re.compile('d_5')
regex_encoded_digit_4 = re.compile('d_4')
regex_encoded_digit_3 = re.compile('d_3')
regex_encoded_digit_2 = re.compile('d_2')
regex_encoded_digit_list = [regex_encoded_digit_2,
                            regex_encoded_digit_3,
                            regex_encoded_digit_4,
                            regex_encoded_digit_5]

regex_digit_reverse = re.compile('p \* 10000 \+ q \* 1000 \+ k \* 100 \+ j \* 10 \+ i|'
                                 'q \* 1000 \+ k \* 100 \+ j \* 10 \+ i|'
                                 'k \* 100 \+ j \* 10 \+ i|'
                                 'j \* 10 \+ i')
regex_digit_reverse_5 = re.compile('p \* 10000 \+ q \* 1000 \+ k \* 100 \+ j \* 10 \+ i|')
regex_digit_reverse_4 = re.compile('q \* 1000 \+ k \* 100 \+ j \* 10 \+ i|')
regex_digit_reverse_3 = re.compile('k \* 100 \+ j \* 10 \+ i|')
regex_digit_reverse_2 = re.compile('j \* 10 \+ i')
regex_digit_reverse_list = [regex_digit_reverse_2,
                            regex_digit_reverse_3,
                            regex_digit_reverse_4,
                            regex_digit_reverse_5]

regex_encoded_digit_reverse = re.compile('dr_5|dr_4|dr_3|dr_2')
regex_encoded_digit_reverse_5 = re.compile('dr_5')
regex_encoded_digit_reverse_4 = re.compile('dr_4')
regex_encoded_digit_reverse_3 = re.compile('dr_3')
regex_encoded_digit_reverse_2 = re.compile('dr_2')
regex_encoded_digit_reverse_list = [regex_encoded_digit_reverse_2,
                                    regex_encoded_digit_reverse_3,
                                    regex_encoded_digit_reverse_4,
                                    regex_encoded_digit_reverse_5]

sort_lambda_expression = 'd.items(), key=lambda item: item[1]'


def simplify_code(code):
    code = code.replace('\'', '"')
    code_part_list = code.split(';')

    code_part_list = [encode(cp) for cp in code_part_list]

    return ' ; '.join(code_part_list)


def restore_code(code):
    code = code.replace('\'', '"')
    code_part_list = code.split(';')

    code_part_list = [decode(cp) for cp in code_part_list]

    return ' ; '.join(code_part_list)


def encode(code):
    # iteration 코드 축약
    code = encode_multiple_iteration(code)
    # digit 표현식 축약
    code = encode_digit_expression(code)
    # sorted 코드 축약
    code = encode_sorted_expression(code)
    return code


def decode(code):
    # iteration 표현식 디코드
    code = decode_multiple_iteration(code)
    # digit 표현식 디코드
    code = decode_digit_expression(code)
    # sorted 표현식 디코드
    code = decode_sorted_expression(code)
    return code


# loop, uniq, nz, asc, desc 등의 expression 에서, 완전한 괄호 안의 expression 을 반환하는 코드
def find_complete_expression(regex, parsed_list):

    if regex not in parsed_list:
        return None

    start_idx = parsed_list.index(regex)

    # bracket stack
    bs = 0
    end_idx = -1
    for idx in range(start_idx + 1, len(parsed_list)):
        if parsed_list[idx] == '(':
            bs += 1
        if parsed_list[idx] == ')':
            bs -= 1
        if bs == 0:
            end_idx = idx + 1
            break

    return ''.join(parsed_list[start_idx:end_idx])


# loop, uniq, nz, asc, desc 등의 expression 에서, 완전한 괄호 안의 expression 을 반환하는 코드
def find_complete_expression_with_index(regex, parsed_list):

    if regex not in parsed_list:
        return None

    start_idx = parsed_list.index(regex)

    # bracket stack
    bs = 0
    end_idx = -1
    for idx in range(start_idx + 1, len(parsed_list)):
        if parsed_list[idx] == '(':
            bs += 1
        if parsed_list[idx] == ')':
            bs -= 1
        if bs == 0:
            end_idx = idx + 1
            break

    return ''.join(parsed_list[start_idx:end_idx]), [start_idx, end_idx]


def encode_sorted_expression(code):
    parsed = print_node(parso.parse(code), list())
    # encoded = ' '.join(parsed)

    if 'sorted' in parsed:
        target_expression = find_complete_expression('sorted', parsed).replace('lambda', ' lambda ')
        inner_parts = target_expression[7:-1].split(',')
        len_in_parsed = len(print_node(parso.parse(target_expression), list()))

        if inner_parts[-1].split('=')[-1] == 'False':
            sort_type = 'asc'
        else:
            sort_type = 'desc'

        # lambda 를 포함한 sorted 인 경우
        if len(inner_parts) == 3:
            target = 'd'
        # 일반적인 sorted 인 경우
        else:
            target = inner_parts[0]

        string = '%s(%s)' % (sort_type, target)

        s_idx = parsed.index('sorted')
        e_idx = s_idx + len_in_parsed - 1

        decode_parsed = print_node(parso.parse(string), list())
        parsed = parsed[0:s_idx] + decode_parsed + parsed[e_idx:]

    return ' '.join(parsed)


def decode_sorted_expression(code):
    parsed = print_node(parso.parse(code), list())

    asc = find_complete_expression('asc', parsed)
    if asc is not None:
        target = asc[4:-1] if asc[4:-1] != 'd' else sort_lambda_expression
        len_in_parsed = len(print_node(parso.parse(asc), list()))
        s_idx = parsed.index('asc')
        e_idx = s_idx + len_in_parsed - 1

        decode_parsed = print_node(parso.parse('sorted(%s,reverse=False)' % target), list())
        parsed = parsed[0:s_idx] + decode_parsed + parsed[e_idx:]

    desc = find_complete_expression('desc', parsed)
    if desc is not None:
        target = desc[5:-1] if desc[5:-1] != 'd' else sort_lambda_expression
        len_in_parsed = len(print_node(parso.parse(desc), list()))
        s_idx = parsed.index('desc')
        e_idx = s_idx + len_in_parsed - 1

        decode_parsed = print_node(parso.parse('sorted(%s,reverse=True)' % target), list())
        parsed = parsed[0:s_idx] + decode_parsed + parsed[e_idx:]

    return ' '.join(parsed)


def encode_multiple_iteration(code):
    parsed = print_node(parso.parse(code), list())
    encoded = ' '.join(parsed)

    if '[' in parsed and 'for' in parsed:
        bracket_indices_in_txt = [encoded.index('['), len(encoded) - list(list(encoded).__reversed__()).index(']')]
        bracket_indices = [parsed.index('['), len(parsed) - list(parsed.__reversed__()).index(']')]
        org = ' '.join(parsed[bracket_indices[0]+1:bracket_indices[1]-1])

        sp_for_only = re.split(' for ', org)
        sp = re.split(' for | if ', org)

        has_const = len(sp) != len(sp_for_only)

        exp_target = sp[0]
        exp_iter = sp[1:-1] if has_const else sp[1:]
        exp_iter_var_list = [i.split(' in ')[0] for i in exp_iter]
        temp_ranges = [i.split(' in ')[1] for i in exp_iter]
        exp_range_list = sorted(set(temp_ranges), key=temp_ranges.index)

        # 모든 range expression 이 겹치는 경우에만 합치고, 하나라도 겹치지 않는 expression 이 있다면 모두 & 로 처리한다.
        if len(exp_range_list) != 1:
            exp_range_list = temp_ranges

        # 다중 for 문의 변수 순서가 맞는지 확인
        if exp_iter_var_list != iter_var_sequence[0:len(exp_iter_var_list)]:
            return encoded

        # constraint expression 과 range expression 을 축약
        exp_const = encode_iteration_constraints(code=sp[-1], iter_count=len(exp_iter)) if has_const else 'True'
        exp_range = ' & '.join([encode_iteration_range(exp) for exp in exp_range_list])

        string = 'loop(%d|%s|%s|%s)' % (len(exp_iter), exp_range, exp_const, exp_target)

        encoded = encoded[0:bracket_indices_in_txt[0]] + string + encoded[bracket_indices_in_txt[1]:]

    return encoded.strip()


def decode_multiple_iteration(code):

    parsed = print_node(parso.parse(code), list())
    decoded = ' '.join(parsed)

    uniq_insertion_needed = False

    mi = find_complete_expression('loop', parsed)
    if mi is None:
        mi = find_complete_expression('loop_uniq', parsed)
        uniq_insertion_needed = True

    if mi is not None:
        elements = mi[mi.index('(')+1:-1].split('|')
        iter_count = int(elements[0])

        if uniq_insertion_needed and iter_count > 1:
            elements[2] = elements[2] + 'and' + 'uniq' if elements[2] != '' else 'uniq'

        var_list = iter_var_sequence[0:iter_count]
        # range_exp = decode_iteration_range(elements[1])
        range_exp = [decode_iteration_range(exp) for exp in elements[1].split('&')]
        if len(range_exp) != len(var_list):
            for i in range(0, len(var_list) - len(range_exp)):
                range_exp.append(range_exp[0])
        mi_exp = ' '.join('for %s in %s' % (var, range_exp[idx]) for idx, var in enumerate(var_list))
        constraints = ' '.join(print_node(parso.parse(elements[2].replace('and', ' and ').replace('or', ' or ')), list()))

        string = '[%s %s if %s]' % (elements[3], mi_exp, decode_iteration_constraints(constraints, iter_count))

        target = ' '.join(print_node(parso.parse(mi.replace('and', ' and ').replace('or', ' or ')), list()))
        dst = ' '.join(print_node(parso.parse(string), list()))
        decoded = decoded.replace(target, dst)

    return decoded


def encode_iteration_constraints(code, iter_count):

    # 조건을 or 기준으로 split 하고, 그 안에서 and 를 합친다.
    const_list = re.split(' or ', code)

    for idx, const in enumerate(const_list):
        c_list = re.split(' and ', const)
        v_list = iter_var_sequence[0:iter_count]

        r_list = list()

        # 유니크 조건 축약
        # not duplicate constraints
        ndc_list = list(set([c for c in c_list if re.sub(regex_not_dup_constraint, '&', c) == '&']))
        # ndc_list 의 갯수가 unique 조건을 만족하면, 축약한다.
        if len(ndc_list) >= 1 and len(ndc_list) == (iter_count*(iter_count-1))/2:
            # unique_exp = 'uniq(%s)' % ','.join(v_list)
            unique_exp = 'uniq'
            r_list.append(unique_exp)
        else:
            r_list += ndc_list

        # 0이 아닌 조건 축약
        # not zero constraints
        nzc_list = list(set([c for c in c_list if re.sub(regex_not_zero_constraint, '&', c) == '&']))
        if len(nzc_list) > 0:
            n_zero_exp = 'nz(%s)' % ','.join([re.sub(' != 0', '', nzc) for nzc in nzc_list])
            r_list.append(n_zero_exp)
        else:
            r_list += nzc_list

        # 축약된 표현 원래 표현식에서 제거
        for d_target in set(ndc_list + nzc_list):
            if d_target in c_list:
                c_list.remove(d_target)

        # 기존 표현 + 축약 표현 merge
        r_list += c_list

        const_list[idx] = ' and '.join(r_list)

    code = ' or '.join(const_list)
    return code


def decode_iteration_constraints(code, iter_count):

    parsed = print_node(parso.parse(code), list())
    decoded = code.replace(' ', '').replace('and', ' and ').replace('or', ' or ')

    # 유니크 조건 디코드
    ndc = 'uniq'
    if ndc in parsed:
        ndc_decoded = ''
        v_list = iter_var_sequence[0:iter_count]
        for idx in range(0, len(v_list) - 1):
            for s_idx in range(idx + 1, len(v_list)):
                if ndc_decoded == '':
                    ndc_decoded = '%s != %s' % (v_list[idx], v_list[s_idx])
                else:
                    ndc_decoded += ' and %s != %s' % (v_list[idx], v_list[s_idx])

        decoded = decoded.replace(ndc, ndc_decoded)

    # 0이 아닌 조건 디코드
    nzc = find_complete_expression('nz', parsed)
    if nzc is not None:
        nzc_decoded = ''
        v_list = nzc[3:-1].split(',')
        for v in v_list:
            if nzc_decoded == '':
                nzc_decoded = '%s != 0' % v
            else:
                nzc_decoded += ' and %s != 0' % v

        decoded = decoded.replace(nzc, nzc_decoded)

    return decoded


def encode_iteration_range(code):

    target = code.replace(' ', '')

    # 예약된 range_expression 인 경우
    if re.sub(regex_reserved_range, '', target) == '':
        code = regex_encoded_reserved_range_list[[idx for idx, regex in enumerate(regex_reserved_range_list) if re.match(regex, target) is not None][0]].pattern

    return code


def decode_iteration_range(code):

    if re.sub(regex_encoded_reserved_range, '', code) == '':
        code = regex_reserved_range_list[[idx for idx, regex in enumerate(regex_encoded_reserved_range_list) if re.match(regex, code) is not None][0]].pattern.split('|')[0].replace('\\', '')

    return code


def encode_digit_expression(code):

    if re.search(regex_digit, code) is not None:
        dl = regex_digit_list
        edl = regex_encoded_digit_list
        t_idx = [idx for idx, regex in enumerate(dl) if re.search(regex, code) is not None][0]
        code = re.sub(dl[t_idx].pattern, edl[t_idx].pattern, code)

    if re.search(regex_digit_reverse, code) is not None:
        dl = regex_digit_reverse_list
        edl = regex_encoded_digit_reverse_list
        t_idx = [idx for idx, regex in enumerate(dl) if re.search(regex, code) is not None][0]
        code = re.sub(dl[t_idx].pattern, edl[t_idx].pattern, code)

    return code


def decode_digit_expression(code):

    if re.search(regex_encoded_digit, code) is not None:
        dl = regex_digit_list
        edl = regex_encoded_digit_list
        t_idx = [idx for idx, regex in enumerate(edl) if re.search(regex, code) is not None][0]
        code = code.replace(edl[t_idx].pattern, dl[t_idx].pattern.replace('\\', ''))

    if re.search(regex_encoded_digit_reverse, code) is not None:
        dl = regex_digit_reverse_list
        edl = regex_encoded_digit_reverse_list
        t_idx = [idx for idx, regex in enumerate(edl) if re.search(regex, code) is not None][0]
        code = code.replace(edl[t_idx].pattern, dl[t_idx].pattern.replace('\\', ''))

    return code


def print_node(node, answer):
    if hasattr(node, 'children'):
        for i in range(len(node.children)):
            answer = print_node(node.children[i], answer)

    else:
        code = node.get_code().replace(" ","")
        answer.append(code)
        return answer

    return answer


def convert_math_method(parsed_list):

    result = list()

    if 'math' not in parsed_list:
        return parsed_list

    skip_index_list = list()

    for i in range(0, len(parsed_list)):

        if i in skip_index_list:
            continue

        p = parsed_list[i]

        if i == len(parsed_list) - 1:
            result.append(p)
            continue

        n_p = parsed_list[i+1]

        if p == 'math' and n_p == '.':
            method = parsed_list[i+2]
            if method == 'perm':
                result.append('func_perm')
                skip_index_list.append(i+1)
                skip_index_list.append(i+2)
            elif method == 'lcm':
                result.append('func_lcm')
                skip_index_list.append(i+1)
                skip_index_list.append(i+2)
            elif method == 'comb':
                result.append('func_comb')
                skip_index_list.append(i+1)
                skip_index_list.append(i+2)
            else:
                result.append(p)
        else:
            result.append(p)

    return result


def convert_custom_method(code):

    parsed = convert_math_method(print_node(parso.parse(code), list()))

    while find_complete_expression_with_index('func_perm', parsed) is not None:
        search = find_complete_expression_with_index('func_perm', parsed)
        exp = search[0]
        exp_elements = exp[10:-1].split(',')
        new_exp = "(math.factorial(%s) // math.factorial((%s)-(%s)))" % (exp_elements[0], exp_elements[0], exp_elements[1])
        parsed = parsed[0:search[1][0]] + print_node(parso.parse(new_exp), list()) + parsed[search[1][1]:]

    while find_complete_expression_with_index('func_lcm', parsed) is not None:
        search = find_complete_expression_with_index('func_lcm', parsed)
        exp = search[0]
        exp_elements = exp[9:-1].split(',')
        new_exp = "(%s*%s//math.gcd(%s,%s))" % (exp_elements[0], exp_elements[1], exp_elements[0], exp_elements[1])
        parsed = parsed[0:search[1][0]] + print_node(parso.parse(new_exp), list()) + parsed[search[1][1]:]

    while find_complete_expression_with_index('func_comb', parsed) is not None:
        search = find_complete_expression_with_index('func_comb', parsed)
        exp = search[0]
        exp_elements = exp[10:-1].split(',')
        new_exp = "(math.factorial(%s) // (math.factorial((%s)-(%s))*math.factorial(%s)))" % (exp_elements[0], exp_elements[0], exp_elements[1], exp_elements[1])
        parsed = parsed[0:search[1][0]] + print_node(parso.parse(new_exp), list()) + parsed[search[1][1]:]

    code = ' '.join(parsed)

    return code


def test(code):
    print("before simplify\n%s" % code)
    simplify_code(code)
    print("after simplify\n%s" % code)
    restore_code(code)
    print("after restore\n%s" % code)
