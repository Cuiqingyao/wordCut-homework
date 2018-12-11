# @File  : cut.py
# @Author: Cuiqingyao
# @Date  : 18-11-29
# @Desc  : 
# @Contact: qingyaocui@gmail.com

from cut_model import Ngram, forward_maximum_match, backward_maximun_match, HMM
from utils.util import writer_answer_to_txt
from settings import RESULT_FILE, TEST_FILE
from tqdm import tqdm
n_gram = Ngram()
hmm_model = HMM()
def compare_p(segments1, segments2):

    p1 = n_gram.cal_sentence_prob(segments1)
    p2 = n_gram.cal_sentence_prob(segments2)
    return segments1 if p1 > p2 else segments2

def cut(sentence):
    segments1 = forward_maximum_match(sentence)
    segments2 =  backward_maximun_match(sentence)
    need_compare = False
    if len(segments1) != len(segments2):
       need_compare = True
       print('cal prob for len not equal')
    else:
        for i in range(len(segments1)):
            if segments1[i] != segments2[i]:
                need_compare = True
                print('cal prob for element not equal')
                break

    if need_compare:
        return compare_p(segments1, segments2)
    else:
        return segments1

def hmm_cut(sentence):
    return hmm_model.viterbi_cut(sentence)

def main(test_file=TEST_FILE, hmm=False):
    results = []
    with open(test_file, 'r', encoding='UTF-8-sig') as f:
        lines = f.readlines()
        for i in tqdm(range(len(lines))):
            if hmm:
                results.append(hmm_cut(lines[i].strip()))
            else:
                results.append(cut("<B>"+lines[i].strip()+"<E>"))

        writer_answer_to_txt(results=results, file=RESULT_FILE)



if __name__ == '__main__':
    main(hmm=True)