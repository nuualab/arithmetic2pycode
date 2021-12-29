# -*- coding: utf-8 -*-
import re
import numpy as np
import torch
from torch.autograd import Variable
# from lib.evaluate import *

class Prefix_To_Infix:
    def __init__ (self):
        self.stack = []

    def push (self, p):
        if p in ['+', '-', '*', '/', '//', '%']:
            op1 = self.stack.pop ()
            op2 = self.stack.pop ()
            self.stack.append ('(%s %s %s)' % (op1, p, op2) )

        elif p in ['math.sqrt']:
            op1 = self.stack.pop()
            op2 = self.stack.pop()
            self.stack.append('%s(%s,%s)' % (p, op1, op2))

        elif p in ['abs']:
            op = self.stack.pop()
            self.stack.append('%s(%s)'%(p, op))
        elif p == '!':
            op = self.stack.pop ()
            self.stack.append ('%s!' % (op) )
        elif p in ['sin', 'cos', 'tan']:
            op = self.stack.pop ()
            self.stack.append ('%s(%s)' % (p, op) )
        else:
            self.stack.append (p)

    def convert (self, l):
        self.stack = []
        reversed_l = reversed(l)
        for e in reversed_l:
            self.push (e)
        return self.stack.pop ()


class Infix_To_Prefix:
    precedence={'math.sqrt':6, 'abs':6, '^':5, '*':4, '/':4, '%':4, '//':4, '+':3,'-':3,'(':2,')':1}
    def __init__(self):
        self.items=[]
        self.size=-1
        self.nums = ['①','②','③','④','⑤','⑥','⑦','⑧','⑨']

    def push(self,value):
        self.items.append(value)
        self.size+=1
    
    def isempty(self):
        if(self.size==-1):
            return True
        else:
            return False
    def seek(self):
        if self.isempty():
            return False
        else:
            return self.items[self.size]
    def is0perand(self, i, nums):
        if i.isalpha() or i in '1234567890' or i in nums:
            return True
        else:
            return False
    def reverse(self,expr):
        rev= []
        for i in expr:
            if i == '(':
                i=')'
            elif i == ')':
                i='('
            rev.insert(0, i)
        return rev
    def infixtoprefix (self, expr, nums):
        prefix=[]
        idx = -1
        # for i in expr:
        op = ['+', '-', '*', '/', '//', '%', '^']

        while idx < len(expr) - 1:
            idx += 1
            i = expr[idx]

            if i == '}':
                temp = idx
                
                while(expr[temp] != '{'):
                    temp += 1

                i = expr[idx:temp+1][::-1]
                prefix.append(i)
                idx = temp
            elif(self.is0perand(i, nums)):
                prefix.append(i)
            elif(i in op):
                while(len(self.items)and self.precedence[i] < self.precedence[self.seek()]):
                    prefix+=self.pop()
                self.push(i)
            elif i == '(':
                self.push(i)
            elif i == ')':
                o=self.pop()
                while o !='(':
                    prefix.append(o)
                    o=self.pop()
                #end of for
        while len(self.items):
            if(self.seek()=='('):
                self.pop()
            else:
                prefix.append(self.pop())
        return prefix

    def pop(self):
        if self.isempty():
            return 0
        else:
            self.size-=1
        
        return self.items.pop()

    def prob_augmented_infixtoprefix (self, expr):
        prefix=[]
        idx = -1
        op = [key for key in self.precedence][:-2]
        # for i in expr:    

        while idx < len(expr) - 1:
            idx += 1
            i = expr[idx]
            try:
                float(i)
                prefix.append(i)
                continue
            except:
                pass

            if i in self.nums:
                prefix.append(i)

            elif i == ',':
                while(len(self.items) and self.precedence[self.seek()] >2):
                    prefix.append(self.pop())

            elif i in op:
                # i = i[::-1]
                if i == '/' and expr[idx+1] == '/':
                    i = '//'
                    idx += 1
                while( len(self.items) and self.precedence[i] < self.precedence[self.seek()]):
                    prefix.append(self.pop())
                
                self.push(i)
            elif i == '(':
                self.push(i)
            elif i == ')':
                o=self.pop()
                while o != '(':
                    prefix.append(o)
                    o=self.pop()
                #end of for

        while len(self.items):
            if(self.seek()=='('):
                self.pop()
            else:
                prefix.append(self.pop())

        return prefix


def get_name_counts(output):
    output = output.replace('"', "'")
    get_counts = False

    if "'" not in output:
        return 0

    else:
        splited_output = output.split(";")
        if splited_output[0][:1] == "d":
            for row in splited_output:
                new_row = row.replace(" ", "")
                if "a=" in new_row and "][0" in new_row:
                    get_counts = True
                    break

            if get_counts:
                for row in splited_output:
                    new_row = row.replace(" ", "")
                    if "d=" in new_row:
                        num = list(new_row).count("'") // 2
                        return num


        elif splited_output[0][:2] == "l0":
            for row in splited_output:
                new_row = row.replace(" ", "")

                if "a=" in new_row and "l0[" in new_row:
                    get_counts = True
                    break
            
            if get_counts:
                for row in splited_output:
                    new_row = row.replace(" ", "")
                    
                    if "l0=" in new_row:
                        num = list(new_row).count("'")
                        return num
    
    return 0


def preprocess_question(questions):
    unit = {"mm"  :  "밀리미터",
            "cm"  :  "센티미터",
            "m"   : "미터",
            "km"  :  "킬로미터",
            "㎟"  :  "제곱밀리미터 ",
            "㎠"  :  "제곱센티미터 ",
            "㎡"  :  "제곱미터",
            "㎢"  :  "제곱킬로미터" ,
            "㎣"  :  "세제곱밀리미터",
            "㎤"  :  "세제곱센티미터",
            "㎥"  :  "세제곱미터" ,
            "㎦"  :  "세제곱키로미터"}

    nums = ['①','②','③','④','⑤','⑥','⑦','⑧','⑨']

    new_questions = []
    for ques in questions:
        ques = list(ques)
        try:
            new_ques = ""
            i = -1
            while i < len(ques) - 1:
                i += 1
                # 퍼센트
                if ques[i] == "%":
                    new_ques += "퍼센트"
                
                
                # 그램
                if ques[i] == "g":
                    new_ques += "그램"
                elif ''.join(ques[i:i+2]) == "kg":
                    i += 1
                    new_ques += "킬로그램"
                    
                # 리터
                if ques[i] == "l":
                    new_ques += "리터"
                elif ''.join(ques[i:i+2]) == "mL":
                    i += 1
                    new_ques += "밀리리터"
                
                # 미터
                elif ''.join(ques[i:i+2]) == "m²":
                    new_ques += "제곱미터"
                    i += 1

                elif ''.join(ques[i:i+2]) == "m³":
                    new_ques += "세제곱미터"
                    i += 1

                elif ''.join(ques[i:i+2]) == "mm":
                    if ''.join(ques[i:i+3]) == "mm²":
                        new_ques += "제곱밀리미터"
                        i += 1

                    elif ''.join(ques[i:i+3]) == "mm³":
                        new_ques += "세제곱밀리미터"
                        i += 1

                    else:
                        new_ques += unit["mm"]

                    i += 1

                elif ques[i] == "c" and ques[i+1] == "m":

                    if ''.join(ques[i:i+3]) == "cm²":
                        new_ques += "제곱센티미터"
                        i += 1
                    elif ''.join(ques[i:i+3]) == "cm³":
                        new_ques += "세제곱센티미터"
                        i += 1

                    else:
                        new_ques += unit["cm"]
                    i += 1

                elif ques[i] == "k" and ques[i+1] == "m":
                    if ''.join(ques[i:i+3]) == "km²":
                        new_ques += "제곱킬로미터"
                        i += 1

                    elif ''.join(ques[i:i+3]) == "km³":
                        new_ques += "세제곱킬로미터"
                        i += 1
                    else:
                        new_ques += unit["km"]
                    i += 1

                elif ques[i] in unit:
                    new_ques += unit[ques[i]]

                else:
                    new_ques += ques[i]

            new_questions.append(new_ques)
            
        except Exception as e:
            new_questions.append(''.join(ques))
        
    return new_questions


def mapping_numbers(outputs, num_dict):
    for output, numbers in zip(outputs, num_dict):
        for i in range(len(output)):
            if output[i] in numbers:
                output[i] = numbers[output[i]]
                
    return outputs


#

