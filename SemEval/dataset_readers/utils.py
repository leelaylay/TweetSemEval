
import pandas as pd
import re
import string

from nltk.corpus import stopwords


# Emoticons
pos_emo = [':-)', ':)', '(-:', '(:', ':-]', '[-:', ':]', '[:', ':-d', ':>)', ':>d', '(<:', ':d', 'b-)', ';-)',
                '(-;', ';)', '(;', ';-d', ';>)', ';>d', '(>;', ';]', '=)', '(=', '=d', '=]', '[=', '(^_^)', '(^_~)',
                '^_^', '^_~', ':->', ':>', '8-)', '8)', ':-}', ':}', ':o)', ':c)', ':^)', '<-:', '<:', '(-8', '(8',
                '{-:', '{:', '(o:', '(^:', '=->', '=>', '=-}', '=}', '=o)', '=c)', '=^)', '<-=', '<=', '{-=', '{=',
                '(o=', '(^=', '8-]', '8]', ':o]', ':c]', ':^]', '[-8', '[8', '[o:', '[^:', '=o]', '=c]', '=^]', '[o=',
                '[^=', '8‑d', '8d', 'x‑d', 'xd', ':-))', '((-:', ';-))', '((-;', '=))', '((=', ':p', ';p', '=p']
neg_emo = ['#-|', ':-&', ':&', ':-(', ')-:', '(t_t)', 't_t', '8-(', ')-8', '8(', ')8', '8o|', '|o8', ':$', ':\'(',
                ':\'-(', ':(', ':-/', ')\':', ')-\':', '):', '\-:', ':\'[', ':\'-[', ':-[', ']\':', ']-\':', ']-:',
                '=-(', '=-/', ')\'=', ')-\'=', ')-=', '\-=', ':-<', ':-c', ':-s', ':-x', ':-|', ':-||', ':/', ':<',
                ':[', ':o', ':|', '=(', '=[', '=\'(', '=\'[', ')=', ']=', '>-:', 'x-:', '|-:', '||-:', '\:', '>:', ']:',
                'o:', '|:', '=|', '=x', 'x=', '|=', '>:(', ':((', '):<', ')):', '>=(', '=((', ')=<', '))=', ':{', ':@',
                '}:', '@:', '={', '=@', '}=', '@=', 'd‑\':', 'd:<', 'd:', 'd8', 'd;', 'd=', 'd‑\'=', 'd=<', 'dx']

DATA_DIR = '../data/GOLD/Subtask_A/'
PROC_DIR = '../data/processed/'
EVAL1_DIR = '../data/Dev/'
EVAL2_DIR = '../data/Final/'

TRAIN = 'twitter-2016train-A.txt'
TEST = 'twitter-2016test-A.txt'
DEV = 'twitter-2016dev-A.txt'
DEVTEST = 'twitter-2016devtest-A.txt'
EVAL1 = 'SemEval2017-task4-dev.subtask-A.english.INPUT.txt'
EVAL2 = 'SemEval2017-task4-test.subtask-A.english.txt'



emoticons_str = r"""
    (?:
        [:=;] # Eyes
        [oO\-]? # Nose (optional)
        [D\)\]\(\]/\\OpP] # Mouth
    )"""

regex_str = [
    emoticons_str,
    r'<[^>]+>',  # HTML tags
    r'(?:@[\w_]+)',  # @-mentions
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)",  # hash-tags
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+',  # URLs

    r'(?:(?:\d+,?)+(?:\.?\d+)?)',  # numbers
    r"(?:[a-z][a-z'\-_]+[a-z])",  # words with - and '
    r'(?:[\w_]+)',  # other words
    r'(?:\S)'  # anything else
]

tokens_re = re.compile(r'(' + '|'.join(regex_str) + ')', re.VERBOSE | re.IGNORECASE)
emoticon_re = re.compile(r'^' + emoticons_str + '$', re.VERBOSE | re.IGNORECASE)




def tokenize(s):
    return tokens_re.findall(s)


def preprocess(s, lowercase=False):
    tokens = tokenize(s)
    if lowercase:
        tokens = [token.lower() if emoticon_re.search(token) else token.lower() for token in tokens]
    return tokens

def main():

    data = {
        'train': DATA_DIR + TRAIN,
        'test': DATA_DIR + TEST,
        'dev': DATA_DIR + DEV,
        'devtest': DATA_DIR + DEVTEST,
        'eval-dev': EVAL1_DIR + EVAL1,
        'eval-final': EVAL2_DIR + EVAL2
        }

    punctuation = list(string.punctuation)
    stop = stopwords.words('english') + punctuation + ['rt', 'via']

    for dataset in data:
        #print(dataset)
        with open(data[dataset], 'r') as dataset_f:
            output_data = []
            for line in dataset_f:
                #print(line.split('\t'))

                info = line.strip().split('\t')
                id, label, text = info[0], info[1], ' '.join(info[2:])

                # process text
                tokens = preprocess(text)

                # remove stopwords and others
                tokens = [term.lower() for term in tokens if term.lower() not in stop]

                # remove hashtags
                tokens = [term for term in tokens if not term.startswith('#')]

                # remove profiles
                tokens = [term for term in tokens if not term.startswith('@')]



                d = {
                    'id': id,
                    'label': label,
                    'text': ' '.join(tokens)
                }
                output_data.append(d)

            df = pd.DataFrame(output_data)
            df.to_csv(PROC_DIR+dataset+'.csv')


if __name__ == '__main__':
    main()