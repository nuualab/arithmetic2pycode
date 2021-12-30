# -*- coding: utf-8 -*-
# +
from io import StringIO
from contextlib import redirect_stdout

from lib.utils import *
from lib.CodeLabelSimplifer import *
from lib.number_extractor import *
from lib.PreProcessorV2 import *


# + endofcell="--"
# -*- coding: utf-8 -*-
import argparse
import logging

import os
import numpy as np
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.core.lightning import LightningModule
from torch.utils.data import DataLoader, Dataset
import transformers
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup

from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm.autonotebook import tqdm
# from post_processing import *
import re
import sys
from io import StringIO
import contextlib
import itertools
import math
import json
import time
os.environ["CUDA_VISIBLE_DEVICES"]="0"
@contextlib.contextmanager
def stdoutIO(stdout=None):
    old = sys.stdout
    if stdout is None:
        stdout = StringIO()
    sys.stdout = stdout
    yield stdout
    sys.stdout = old


parser = argparse.ArgumentParser(description='Simsimi based on KoGPT-2')



parser.add_argument('--problem_path',
                    type=str,
                    default='"../inference_results/problem_400.json"',
                    help='model binary for starting chat')


parser.add_argument('--model_params',
                    type=str,
                    default='model_chp/model_-last.ckpt',
                    help='model binary for starting chat')

parser.add_argument('--train',
                    action='store_true',
                    default=False,
                    help='for training')

logger = logging.getLogger()
logger.setLevel(logging.INFO)

U_TKN = '<usr>'
S_TKN = '<sys>'
BOS = '</s>'
EOS = '</s>'
MASK = '<unused0>'
SENT = '<unused1>'
PAD = '<pad>'

# # +

# -

TOKENIZER = PreTrainedTokenizerFast.from_pretrained("./weights/tokenizer/",
            bos_token=BOS, eos_token=EOS, unk_token='<unk>',
            pad_token=PAD, mask_token=MASK)




class NLPdataset(Dataset):
    
    def __init__(self, file, tokenizer, max_len=32, pad_index=0, ignore_index=-100):
        self.document = file
        self.tok = tokenizer
        self.max_len = max_len
        self.pad_index = pad_index
        self.ignore_index = ignore_index
        
    def __len__(self):
        return len(self.document)
   
    def add_padding_data(self, inputs):
        if len(inputs) < self.max_len:
            pad = np.array([self.pad_index] *(self.max_len - len(inputs)))
            inputs = np.concatenate([inputs, pad])
        else:
            inputs = inputs[:self.max_len]

        return inputs

    def add_ignored_data(self, inputs):
        if len(inputs) < self.max_len:
            pad = np.array([self.ignore_index] *(self.max_len - len(inputs)))
            inputs = np.concatenate([inputs, pad])
        else:
            inputs = inputs[:self.max_len]

        return inputs
    
    def __getitem__(self, idx):
        question = self.document[idx]['question']
        input_ids = self.tok.encode(question)
        input_ids = self.add_padding_data(input_ids)
        
        return {'input_ids': np.array(input_ids, dtype=np.int_)}


class NLPmodel(nn.Module):
    
    def __init__(self, electra):
        super(NLPmodel, self).__init__()
        self.model = electra     
        self.linear = nn.Linear(768,1)
    
    def forward(self, x):
        x = self.model(x)
        y = x[0][:,0,:]
        cls = y
        out = self.linear(cls)
        
        return out


class KoGPT2Chat(LightningModule):
    def __init__(self, hparams, **kwargs):
        super(KoGPT2Chat, self).__init__()
        self.hparams = hparams
        self.neg = -1e18
        self.kogpt2 = GPT2LMHeadModel.from_pretrained('./weights/kogpt2/')
        self.loss_function = torch.nn.CrossEntropyLoss(reduction='none')

    @staticmethod
    def add_model_specific_args(parent_parser):
        # add model specific args
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--max-len',
                            type=int,
                            default=32,
                            help='max sentence length on input (default: 32)')

        parser.add_argument('--batch-size',
                            type=int,
                            default=64,
                            help='batch size for training (default: 96)')
        parser.add_argument('--lr',
                            type=float,
                            default=5e-5,
                            help='The initial learning rate')
        parser.add_argument('--warmup_ratio',
                            type=float,
                            default=0.1,
                            help='warmup ratio')
        return parser

    def forward(self, inputs):
        # (batch, seq_len, hiddens)
        output = self.kogpt2(inputs, return_dict=True)
        return output.logits

    def training_step(self, batch, batch_idx):
        token_ids, mask, label = batch
        out = self(token_ids)
        mask_3d = mask.unsqueeze(dim=2).repeat_interleave(repeats=out.shape[2], dim=2)
        mask_out = torch.where(mask_3d == 1, out, self.neg * torch.ones_like(out))
        loss = self.loss_function(mask_out.transpose(2, 1), label)
        loss_avg = loss.sum() / mask.sum()
        self.log('train_loss', loss_avg)
        return loss_avg

    def configure_optimizers(self):
        # Prepare optimizer
        param_optimizer = list(self.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.hparams.lr, correct_bias=False)
        # warm up lr
        num_train_steps = len(self.train_dataloader()) * self.hparams.max_epochs
        num_warmup_steps = int(num_train_steps * self.hparams.warmup_ratio)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps)
        lr_scheduler = {'scheduler': scheduler, 'name': 'cosine_schedule_with_warmup',
                        'monitor': 'loss', 'interval': 'step',
                        'frequency': 1}
        return [optimizer], [lr_scheduler]

    def _collate_fn(self, batch):
        data = [item[0] for item in batch]
        mask = [item[1] for item in batch]
        label = [item[2] for item in batch]
        return torch.LongTensor(data), torch.LongTensor(mask), torch.LongTensor(label)

    def train_dataloader(self):
        data = "problemsheet.json" 
        self.train_set = CharDataset(data, max_len=self.hparams.max_len)
        train_dataloader = DataLoader(
            self.train_set, batch_size=self.hparams.batch_size, num_workers=2,
            shuffle=True, collate_fn=self._collate_fn)
        return train_dataloader

    def test(self, question):
        tok = TOKENIZER
        sent = SENT
        sent_tokens = tok.tokenize(sent)
        answers = []
        with torch.no_grad(): 
            a = ''
            q = question
            
            while 1:
                input_ids = torch.LongTensor(tok.encode(U_TKN + q + SENT + sent + S_TKN + a)).unsqueeze(dim=0)
                pred = self(input_ids.cuda())
                gen = tok.convert_ids_to_tokens(
                            torch.argmax(
                                pred,
                                dim=-1).squeeze().cpu().numpy().tolist())[-1]
                if gen != '_':
                    if gen.replace('_', ' ') != '<pad>' and gen.replace('_', ' ') != '</s>' and gen.replace('_', ' ') != '<pad>' and gen.replace('_', ' ') != '_':
                        if gen.replace('▁', '') != '':
                            answers.append(gen.replace('▁', ''))
                    if gen == EOS:
                        break
                    a += gen.replace('▁', ' ')

                if len(a) > 500:
                    break
                
        return answers
    
    def test2(self, question):
        tok = TOKENIZER
        sent = SENT
        sent_tokens = tok.tokenize(sent)
        answers = []
        with torch.no_grad(): 
            a = ''
            q = question
            while 1:
                input_ids = torch.LongTensor(tok.encode(U_TKN + q + SENT + sent + S_TKN + a)).unsqueeze(dim=0)
                pred = self(input_ids.cuda())
                gen = tok.convert_ids_to_tokens(
                            torch.argmax(
                                pred,
                                dim=-1).squeeze().cpu().numpy().tolist())[-1]
                if gen.replace('_', ' ') != '<pad>' and gen.replace('_', ' ') != '</s>' and gen.replace('_', ' ') != '<pad>' and gen.replace('_', ' ') != '_':
                    answers.append(gen.replace('▁', ''))
                if gen == EOS:
                    break
                a += gen.replace('▁', ' ')
                if len(a) > 500:
                    break
                
        return answers

token_dict = {}
for i in range(70):
    u = '<unused' + str(i+2) + '>'
    token_dict[u] = 't' + str(i+1)

def model2token(answers):
    token = []
    for ans in answers:
        if ans in token_dict:
            token.append(token_dict[ans])
        else:
            token.append(ans)
            
    return token

def remove_n(word):
    index = {}
    index['t'] = []
    index['n'] = []
    index['m'] = []
    index['o'] = []
    num_word = []
    num = []
    for i in range(len(word)):
        w = word[i]
        l = len(index['n'])
        l2 = len(index['t'])
        if w == 'n':
            index['n'].append(i)
        try:
            a = w[0]
            if a == 't':
                index['t'].append(i) 
        except: pass

        if l != len(index['n']) or l2 != len(index['t']):
            num_word = []
        if num != num_word or i == len(word)-1:
            index['m'].append(num)    
        if len(re.compile("^[\-]*[0-9\.\/]+").findall(w)):
            a = w #int(w)
            num_word.append(i)       
        else: 
            if w != 'n':
                index['o'].append(i)
        num = num_word
    new_word = []
    ii = 0
    for i in range(len(word)):
        if i not in index['n']:
            if ii < len(index['m']):
                if len(index['m'][ii]) != 0:
                    if i == index['m'][ii][0]:
                        mwp = ""
                        for j in range(len(index['m'][ii])):
                            mwp += word[index['m'][ii][j]]
                        new_word.append(mwp)
                        ii += 1


            if i in index['o']:
                new_word.append(word[i])
                
    return new_word

def remove_w(token):
    w_dict = {}
    w_dict['w'] = []
    w_dict['h'] = []
    w_dict['n'] = []
    word = []
    num = []
    new_word = []
    for i in range(len(token)):
        t = token[i]
        length = len(w_dict['w'])
        if t == 'w':
            w_dict['w'].append(i)
        if length != len(w_dict['w']):
            word = []
        if num != word or i == len(token)-1:
            w_dict['h'].append(num)

        if len(re.compile('[\)\(|가-힣]').findall(str(t))) != 0:
            word.append(i)
        if t != 'w' and len(re.compile('[\)\(|가-힣]').findall(str(t))) == 0:
            w_dict['n'].append(i)
        num = word


    ii = 0
    for i in range(len(token)):
        if i not in w_dict['w']:
            if ii < len(w_dict['h']):
                if len(w_dict['h'][ii]) != 0:
                    if i == w_dict['h'][ii][0]:
                        mwp = ""
                        for j in range(len(w_dict['h'][ii])):
                            mwp += token[w_dict['h'][ii][j]]
                        new_word.append(mwp)
                        ii +=1
            if i in w_dict['n']:
                new_word.append(token[i])
                
    return new_word


def arithmetic_processing(eq):
    semi = ""
    for w in eq:
        semi += w
    answer = ""
    semi = semi.split(",")
    candi = ""
    
    for i in range(len(semi)):
        word = semi[i]
        t = 100
        for ii in range(len(word)):
            w = word[ii]
            candi += w
            if w == '=':
                candi += 'abs('
                t = i
            if ii == len(word)-1 and t != 100:            
                candi += ')'
        if i != len(semi)-1:
            candi += ','
            
    candi = candi.split(",")
    for word in candi:   
        if word != "":
            if word[0] != ',':      
                if word[0] == '?':
                    string = f"x={word[1:]}\ntry:\n    if int(float(x)) == float(x):\n        x = str(int(float(x)))\n    else:\n        x = str(round(float(x),2))\nexcept:\n    pass\nprint(x)"
                    answer += string
                    #answer += 'print('
                    #answer += 'x'
                    #answer += ')'               
                else:
                    answer += word
                    answer +="\n"
            
    return answer

def make_int(token):
    result = []
    for t in token:
        try:
            result.append(float(t))
        except:
            result.append(t)
            
    return result


def change_eval(question):
    pattern = re.compile("[\ A-Z0-9\+\-\*\/]+[\>\<\=]+[\ A-Z0-9\+\-\*\/]+[\>\<\=]*[\ A-Z0-9\+\-\*\/]*[\>\<\=]*[\ A-Z0-9\+\-\*\/]*[\>\<\=]*[\ A-Z0-9\+\-\*\/]*[\>\<\=]*[\ A-Z0-9\+\-\*\/]*")
    all_data = []
    pattern2 = re.compile("[A-Z]+")
    while re.search(pattern, question):
        m = re.search(pattern,question)
        equation = m.group()
        eval_eq = " 1="+''.join(pattern2.findall(equation))
        eval_question = question[:m.start(0)]+eval_eq
        all_data.append(eval_question)
        question = question[m.end(0):]
    all_data.append(question)
    eval_question_all = ''.join(all_data)
    return eval_question_all

def each_different_processing(output):
    try:
        for i in range(len(output)):
            if "loop" == output[i] and "uniq" not in output:
                if int(output[i+2]) > 1:
                    output[i] = "loop_uniq"

        return output
    
    except:
        return output

def inference_both(outputs, keyword_checks):
    prefix_to_infix = Prefix_To_Infix()
    
    results = []
    temp = []
    for i, output in enumerate(outputs):
        result = {}
        result['answer'] = ""
        result['equation'] = ""
        result['type'] = ""
        result['output'] = output
        code = True
        
        formatted_equation = ""
        formatted_equation += "import math\n"
        formatted_equation += "import itertools\n"
        decoded_return_value = -111
        
        keywords = keyword_checks[i]

        if output[-1] == 'a':
            try:
                if "서로 다른" in keywords:
                    output = each_different_processing(output)
                
                output = convert_custom_method(''.join(output))
                decoded_output = restore_code(output)
                
                result['equation'] = decoded_output
                result['type'] = "c"
                
                name_counts = get_name_counts(decoded_output)

                    # 라벨, 리스트, 문제, 숫자
                
                
                code_lines = decoded_output.split(";")
                for i, line in enumerate(code_lines[:-1]):
                    a = line.strip()
                    formatted_equation += a + "\n"
                # 정답 string인 경우 라벨 꼭 바꾸기!!
                if name_counts == 0:
                    formatted_equation += 'a=abs(a)\n'
                    formatted_equation += "s = '%d' % (int(float(a))) if int(float(a)) == float(a) else '%.2f' % round(float(a),2)\nprint(s)"
                    
                else:
                    formatted_equation += 'print(a)'

                result['equation'] = formatted_equation
                ex = ""

                try :
                    f = StringIO()
                    with redirect_stdout(f):
                        exec(formatted_equation, globals())
                    decoded_return_value = f.getvalue()
                    result['answer'] = decoded_return_value
                    
                except Exception as e:     
                    code = False
                    decoded_return_value = ""
                        
            except Exception as e:
                code = False

        elif output[-1] != 'a' or not code or True:
            try:
                infix_output = prefix_to_infix.convert(output)
                infix_return_value = ""
                result['equation'] = infix_output
                result['type'] = "a"
                
                formatted_equation += 'a=' + infix_output + '\n'
                formatted_equation += 'a=abs(a)\n'
                formatted_equation += "s = '%d' % (int(float(a))) if int(float(a)) == float(a) else '%.2f' % round(float(a),2)\nprint(s)"
                
                
                result['equation'] = formatted_equation
                
                try :
                    f = StringIO()
                    with redirect_stdout(f):
                        exec(formatted_equation, globals())
                    infix_return_value = f.getvalue()
                    result['answer'] = infix_return_value
                    
                except Exception as e:  
                    pass
            except:
                pass
        results.append(result)
    return results

def inference_a(outputs):
    prefix_to_infix = Prefix_To_Infix()

    results = []
    for i, output in enumerate(outputs):
        result = {}
        result['answer'] = ""
        result['equation'] = ""
        result['type'] = ""
        result['output'] = output
        
        formatted_equation = ""
        formatted_equation += "import math\n"
        formatted_equation += "import itertools\n"
        
        try:
            infix_output = prefix_to_infix.convert(output)
            result['type'] = "a"
            
            formatted_equation += 'a=' + infix_output + '\n'
            formatted_equation += 'a=abs(a)\n' 
            formatted_equation += "s = '%d' % (int(float(a))) if int(float(a)) == float(a) else '%.2f' % round(float(a),2)\nprint(s)"

            result['equation'] = formatted_equation
            
            try :
                f = StringIO()
                with redirect_stdout(f):
                    exec(formatted_equation, globals())
                infix_return_value = f.getvalue()
                result['answer'] = infix_return_value

            except Exception as e:  
                pass
        except:
            pass
        
        results.append(result)

    return results


parser = KoGPT2Chat.add_model_specific_args(parser)
parser = Trainer.add_argparse_args(parser)
args = parser.parse_args()
logging.info(args)

mapping_key = ["①","②","③","④","⑤","⑥","⑦","⑧","⑨","ⓐ","ⓑ","ⓒ",
"v0","v1","v2","v3","v4","!=",
"1000","10000","100000","1000000","100000000","1000000000000","3.141592",
"<=","==",">=","EOS","PAD","SOS","True","UNK","abs","asc",
"d_2","d_3","d_4","d_5","desc","dr_2",
"else","for","l0","l1","l2","l3","len","list","loop",
"math.ceil","math.comb","math.fabs","math.factorial","math.gcd","math.lcm","math.log","math.perm","math.sqrt","max","min","nz","pow",
"r_0_10","r_100_1000","r_10_100","range","round","set","str","sum"," uniq "," and "," if ","in"," or "
]
space_words = ["and", "uniq", "if", "or"]

trans_gpt_convert_key = {}
gpt_trans_convert_key = {}
for i, key in enumerate(mapping_key):
    idx = i+10
    value = "<unused" + str(idx) + ">"
    trans_gpt_convert_key[key] = value
    gpt_trans_convert_key[value] = key

question_json = args.problem_path

if __name__ == "__main__":
    
    both_model = KoGPT2Chat.load_from_checkpoint("weights/both.ckpt").cuda()
    a_model = KoGPT2Chat.load_from_checkpoint("weights/a.ckpt").cuda()
    ensemble = True
    
    with open(question_json, "r", encoding='utf-8-sig') as f:
        dat = json.load(f)
        
    raw_questions = []
    # keywords = ['서로 다른']
    # for key in dat:
    #     raw_questions.append(dat[key]['question'])
        
        
    # get question number
    num_dict = []
    keyword_checks = []
    questions = []
    for question in raw_questions:
        keyword_temp = []
        question, _, numbers = preprocess(question, "", {})

        num_dict.append(numbers)
        questions.append(question)
        
        for keyword in keywords:
            if keyword in question:
                keyword_temp.append(keyword)
                
        keyword_checks.append(keyword_temp)
        
    
    operand = ["①",  "②",  "③",  "④",  "⑤",  "⑥",  "⑦",  "⑧",  "⑨"]
    operand_token = ["<unused10>",  "<unused11>",  "<unused12>",  "<unused13>",  "<unused14>",  "<unused15>",  "<unused16>",  "<unused17>",  "<unused18>"]
    
    operand_dict = {}
    for i in range(len(operand)):
        operand_dict[operand[i]] = operand_token[i]
    
    new_num_dicts = []
    for i, question in enumerate(questions):
        new_dict = {}
        for j in range(len(operand)):
            question = question.replace(operand[j], operand_token[j])
        questions[i] = question
        new_num_dicts.append({operand_dict[key]: num_dict[i][key] for key in num_dict[i]})
        

    both_outputs = []
    for i, question in enumerate(questions):

        output = both_model.test(question)
        for j in range(len(output)):
            if output[j] in gpt_trans_convert_key:
                output[j] = gpt_trans_convert_key[output[j]]

            if output[j] in num_dict[i]:
                output[j] = num_dict[i][output[j]]

        both_outputs.append(output)

    both_results = inference_both(both_outputs, keyword_checks)
    
    
    if ensemble:
        a_outputs = []
        for i, question in enumerate(questions):

            output = a_model.test(question)
            for j in range(len(output)):
                if output[j] in gpt_trans_convert_key:
                    output[j] = gpt_trans_convert_key[output[j]]

                if output[j] in num_dict[i]:
                    output[j] = num_dict[i][output[j]]

            a_outputs.append(output)

        a_results = inference_a(a_outputs)
        
        results = []
        
        for i in range(len(both_results)):
            if both_results[i]["type"] == 'c':
                results.append(both_results[i])
                
            elif a_results[i]["answer"] != "" or (a_results[i]["answer"] == "" and both_results[i]['answer'] == ""):
                results.append(a_results[i])
                
            else:
                results.append(both_results[i])
                
        
    else:
        results = both_results
        
    json_result = {}
    for idx, result in enumerate(results):
        
        equation = result['equation']
        answer = result['answer']
        
        json_result[idx+1] = {}
#         json_result[idx+1]['answer'] = answer.strip()
#         json_result[idx+1]['equation'] = equation

        if answer != "":
            json_result[idx+1]['answer'] = answer.strip()
            json_result[idx+1]['equation'] = equation
        else:
            json_result[idx+1]['answer'] = ""
            json_result[idx+1]['equation'] = ""

    with open("../inference_results/a_15_both_7_test.json", "w", encoding="UTF-8") as f:    
        json.dump(json_result, f, ensure_ascii=False, indent=4)
# --




