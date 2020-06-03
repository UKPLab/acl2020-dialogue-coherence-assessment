import math
import os
from copy import deepcopy
from ast import literal_eval
import pandas as pd
from math import factorial
import random
from collections import Counter, defaultdict
import sys
from nltk import word_tokenize
from tqdm import tqdm, trange
import argparse
import numpy as np
import re
import csv
from sklearn.model_selection import train_test_split

from swda.swda import CorpusReader, Transcript, Utterance

act2word = {1:"inform",2:"question", 3:"directive", 4:"commissive"}

def permute(sents, sent_DAs, amount):
    """ return a list of different! permuted sentences and their respective dialog acts """
    """ if amount is greater than the possible amount of permutations, only the uniquely possible ones are returned """
    assert len(sents) == len(sent_DAs), "length of permuted sentences and list of DAs must be equal"

    if amount == 0:
        return []

    permutations = [list(range(len(sents)))]
    amount = min(amount, factorial(len(sents))-1)
    for i in range(amount):
        permutation = np.random.permutation(len(sents))
        while permutation.tolist() in permutations:
            permutation = np.random.permutation(len(sents))

        permutations.append(permutation.tolist())
    return permutations[1:] #the first one is the original, which was included s.t. won't be generated

def draw_rand_sent(act_utt_df, sent_len, amount):
    """ df is supposed to be a pandas dataframe with colums 'act' and 'utt' (utterance), 
        with act being a number from 1 to 4 and utt being a sentence """

    permutations = []
    for _ in range(amount):
        (utt, da, name, ix) = draw_rand_sent_from_df(act_utt_df)
        sent_insert_ix = random.randint(0, sent_len-1)
        permutations.append((utt, da, name, ix, sent_insert_ix))
    return permutations

def draw_rand_sent_from_df(df):
    ix = random.randint(0, len(df['utt'])-1)
    return literal_eval(df['utt'][ix]), df['act'][ix], df['dialogue'][ix], df['ix'][ix]

def half_perturb(sents, sent_DAs, amount):
    assert len(sents) == len(sent_DAs), "length of permuted sentences and list of DAs must be equal"

    permutations = [list(range(len(sents)))]

    for _ in range(amount):
        while True:
            speaker = random.randint(0,1) # choose one of the speakers
            speaker_ix = list(filter(lambda x: (x-speaker) % 2 == 0, range(len(sents))))

            permuted_speaker_ix = np.random.permutation(speaker_ix)
            new_sents = list(range(len(sents)))
            for (i_to, i_from) in zip(speaker_ix, permuted_speaker_ix):
                new_sents[i_to] = i_from

            if (not new_sents == permutations[0]) and (
                    not new_sents in permutations or len(permutations) > math.factorial(len(speaker_ix))):
                permutations.append(new_sents)
                break

    return permutations[1:]

def utterance_insertions(length, amount):
    possible_permutations = []
    original = list(range(length))
    for ix in original:
        for y in range(length):
            if ix == y: continue

            ix_removed = original[0:ix] + ([] if ix == length-1 else original[ix+1:])
            ix_removed.insert(y, ix)
            possible_permutations.append(deepcopy(ix_removed))

    permutations = []
    for _ in range(amount):
        i = random.randint(0, len(possible_permutations)-1)
        permutations.append(possible_permutations[i])

    return permutations

class DailyDialogConverter:
    def __init__(self, data_dir, tokenizer, word2id, task='', ranking_dataset = True):
        self.data_dir = data_dir
        self.act_utt_file = os.path.join(data_dir, 'act_utt_name.txt')

        self.tokenizer = tokenizer
        self.word2id = word2id
        self.output_file = None
        self.task = task
        self.ranking_dataset = ranking_dataset
        self.perturbation_statistics = 0

        self.setname = os.path.split(data_dir)[1]
        assert self.setname == 'train' or self.setname == 'validation' or self.setname == 'test', "wrong data dir name"

    def create_act_utt(self):
        dial_file = os.path.join(self.data_dir, "dialogues_{}.txt".format(self.setname))
        act_file = os.path.join(self.data_dir, "dialogues_act_{}.txt".format(self.setname))
        output_file = os.path.join(self.data_dir, 'act_utt_name.txt'.format(self.task))

        df = open(dial_file, 'r')
        af = open(act_file, 'r')
        of = open(output_file, 'w')
        csv_writer = csv.writer(of, delimiter='|')

        for line_count, (dial, act) in tqdm(enumerate(zip(df, af)), total=11118):
            seqs = dial.split('__eou__')
            seqs = seqs[:-1]

            if len(seqs) < 5:
                continue

            tok_seqs = [self.tokenizer(seq) for seq in seqs]
            tok_seqs = [[w.lower() for w in utt] for utt in tok_seqs]
            tok_seqs = [self.word2id(seq) for seq in tok_seqs]

            acts = act.split(' ')
            acts = acts[:-1]
            acts = [int(act) for act in acts]

            for utt_i, (act, utt) in enumerate(zip(acts, tok_seqs)):
                dialog_name = "{}_{}".format(self.setname, line_count)
                row = (act, utt, dialog_name,utt_i)
                csv_writer.writerow(row)

    def convert_dset(self, amounts):
        # data_dir is supposed to be the dir with the respective train/test/val-dataset files
        print("Creating {} perturbations for task {}".format(amounts, self.task))

        dial_file = os.path.join(self.data_dir, "dialogues_{}.txt".format(self.setname))
        act_file = os.path.join(self.data_dir, "dialogues_act_{}.txt".format(self.setname))
        self.output_file = os.path.join(self.data_dir, 'coherency_dset_{}.txt'.format(self.task))

        root_data_dir = os.path.split(self.data_dir)[0]
        shuffled_path = os.path.join(root_data_dir, "shuffled_{}".format(self.task))
        if not os.path.isdir(shuffled_path):
            os.mkdir(shuffled_path)

        assert os.path.isfile(dial_file) and os.path.isfile(act_file), "could not find input files"
        assert os.path.isfile(self.act_utt_file), "missing act_utt.txt in data_dir"

        with open(self.act_utt_file, 'r') as f:
            act_utt_df = pd.read_csv(f, sep='|', names=['act','utt','dialogue','ix'])

        rand_generator = lambda: draw_rand_sent_from_df(act_utt_df)

        df = open(dial_file, 'r')
        af = open(act_file, 'r')
        of = open(self.output_file, 'w')

        discarded = 0

        for line_count, (dial, act) in tqdm(enumerate(zip(df, af)), total=11118):
            seqs = dial.split('__eou__')
            seqs = seqs[:-1]

            if len(seqs) < 5:
                discarded += 1
                continue

            tok_seqs = [self.tokenizer(seq) for seq in seqs]
            tok_seqs = [[w.lower() for w in utt] for utt in tok_seqs]
            tok_seqs = [self.word2id(seq) for seq in tok_seqs]

            acts = act.split(' ')
            acts = acts[:-1]
            acts = [int(act) for act in acts]

            if self.task == 'up':
                permuted_ixs = permute(tok_seqs, acts, amounts)
            elif self.task == 'us':
                permuted_ixs = draw_rand_sent(act_utt_df, len(tok_seqs), amounts)
            elif self.task == 'hup':
                permuted_ixs = half_perturb(tok_seqs, acts, amounts)
            elif self.task == 'ui':
                permuted_ixs = utterance_insertions(len(tok_seqs), amounts)

            shuffle_file = os.path.join(shuffled_path, "{}_{}.csv".format(self.setname, line_count))
            with open(shuffle_file, "w") as f:
                csv_writer = csv.writer(f)
                for perm in permuted_ixs:
                    if self.task == 'us':
                        (utt, da, name, ix, insert_ix) = perm
                        row = [name, ix,insert_ix]
                        csv_writer.writerow(row)
                    else:
                        csv_writer.writerow(perm)

            self.perturbation_statistics += len(permuted_ixs)

            if self.task == 'us':
                for p in permuted_ixs:
                    (insert_sent, insert_da, name, ix, insert_ix) = p
                    a = " ".join([str(a) for a in acts])
                    u = str(tok_seqs)
                    p_a = deepcopy(acts)
                    p_a[insert_ix] = insert_da
                    pa = " ".join([str(a) for a in p_a])
                    p_u = deepcopy(tok_seqs)
                    p_u[insert_ix] = self.word2id(insert_sent)
                    of.write("{}|{}|{}|{}|{}\n".format("0",a,u,pa,p_u))
                    of.write("{}|{}|{}|{}|{}\n".format("1",pa,p_u,a,u))

            else:
                for p in permuted_ixs:
                    a = " ".join([str(a) for a in acts])
                    u = str(tok_seqs)
                    pa = [acts[i] for i in p]
                    p_a = " ".join([str(a) for a in pa])
                    pu = [tok_seqs[i] for i in p]
                    p_u = str(pu)
                    of.write("{}|{}|{}|{}|{}\n".format("0",a,u,p_a,p_u))
                    of.write("{}|{}|{}|{}|{}\n".format("1",p_a,p_u,a,u))

        print(discarded)

class SwitchboardConverter:
    def __init__(self, data_dir, tokenizer, word2id, task='', seed=42):
        self.corpus = CorpusReader(data_dir)
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.word2id = word2id
        self.task = task

        self.utt_num = 0
        for utt in self.corpus.iter_utterances():
            self.utt_num += 1

        self.trans_num = 0
        for trans in self.corpus.iter_transcripts():
            self.trans_num += 1

        self.da2num = switchboard_da_mapping()
        
        # CAUTION: make sure that for each task the seed is the same s.t. the splits will be the same!
        train_ixs, val_ixs = train_test_split(range(self.trans_num), shuffle=True, train_size=0.8, random_state=seed)
        val_ixs, test_ixs = train_test_split(val_ixs, shuffle=True, train_size=0.5, random_state=seed)
        self.train_ixs, self.val_ixs, self.test_ixs = train_ixs, val_ixs, test_ixs

        self.utt_da_pairs = []
        prev_da = "%"
        for i, utt in enumerate(self.corpus.iter_utterances()):
            sentence = re.sub(r"([+/\}\[\]]|\{\w)", "",
                            utt.text)

            sentence = self.word2id(self.tokenizer(sentence))
            act = utt.damsl_act_tag()
            if act == None: act = "%"
            if act == "+": act = prev_da

            _, swda_name = os.path.split(utt.swda_filename)
            swda_name = swda_name[:-4] if swda_name.endswith('.csv') else swda_name

            ix = utt.utterance_index

            self.utt_da_pairs.append((sentence, act, swda_name, ix))

    def draw_rand_sent(self):
        r = random.randint(0, len(self.utt_da_pairs)-1)
        return self.utt_da_pairs[r]

    def create_vocab(self):
        print("Creating Vocab file for Switchboard")

        cnt = Counter()
        for utt in self.corpus.iter_utterances():
            sentence = re.sub(r"([+/\}\[\]]|\{\w)", "",
                            utt.text)
            sentence = self.tokenizer(sentence)
            for w in sentence:
                cnt[w] += 1

        itos_file = os.path.join(self.data_dir, "itos.txt")
        itosf = open(itos_file, "w")

        for (word, _) in cnt.most_common(25000):
            itosf.write("{}\n".format(word))


    #getKeysByValue
    def swda_permute(self, sents, amount, speaker_ixs):
        if amount == 0:
            return []

        permutations = [list(range(len(sents)))]
        segment_permutations = []
        amount = min(amount, factorial(len(sents))-1)
        segm_ixs = self.speaker_segment_ixs(speaker_ixs)
        segments = list(set(segm_ixs.values()))

        for i in range(amount):
            while True:
                permutation = []
                segm_perm = np.random.permutation(len(segments))
                segment_permutations.append(segm_perm)
                for segm_ix in segm_perm:
                    utt_ixs = sorted(getKeysByValue(segm_ixs, segm_ix))
                    permutation = permutation + utt_ixs

                if permutation not in permutations:
                    break

            permutations.append(permutation)
        return permutations[1:] , segment_permutations #the first one is the original, which was included s.t. won't be generated

    def speaker_segment_ixs(self, speaker_ixs):
        i = 0
        segment_indices = dict()
        prev_speaker = speaker_ixs[0]
        for j,speaker in enumerate(speaker_ixs):
            if speaker != prev_speaker:
                prev_speaker = speaker
                i += 1
            segment_indices[j] = i
        return segment_indices

    def swda_half_perturb(self, amount, speaker_ixs):
        segm_ixs = self.speaker_segment_ixs(speaker_ixs)
        segments = list(set(segm_ixs.values()))
        segment_permutations = []
        permutations = [list(segm_ixs.keys())]
        for _ in range(amount):
            speaker = random.randint(0,1) # choose one of the speakers
            speaker_to_perm = list(filter(lambda x: (x-speaker) % 2 == 0, segments))
            speaker_orig = list(filter(lambda x: (x-speaker) % 2 != 0, segments))
            #TODO: rename either speaker_ix or speaker_ixs, they are something different, but the names are too close
            if len(speaker_to_perm) < 2:
                return []

            while True:
                permuted_speaker_ix = np.random.permutation(speaker_to_perm).tolist()

                new_segments = [None]*(len(speaker_orig)+len(permuted_speaker_ix))
                if speaker == 0 : 
                    new_segments[::2] = permuted_speaker_ix
                    new_segments[1::2] = speaker_orig
                else:
                    new_segments[1::2] = permuted_speaker_ix
                    new_segments[::2] = speaker_orig
                segment_permutations.append(new_segments)

                permutation = []
                for segm_ix in new_segments:
                    utt_ixs = sorted(getKeysByValue(segm_ixs, segm_ix))
                    permutation = permutation + utt_ixs

                if not permutation in permutations:
                    permutations.append(permutation)
                    break

        return permutations[1:], segment_permutations

    def swda_utterance_insertion(self, speaker_ixs, amounts):
        segment_ixs = self.speaker_segment_ixs(speaker_ixs)
        segments = list(set(segment_ixs.values()))
        segment_permutations = []
        permutations = []

        i = 0
        for _ in range(amounts):
            while True: # actually: do ... while permutation not in permutations
                i_from = random.randint(0, len(segments)-1)
                i_to = random.randint(0, len(segments)-2)
                segm_perm = deepcopy(segments)
                rem_elem = segments[i_from]
                segm_perm = segm_perm[0:i_from] + segm_perm[i_from+1:]
                segm_perm = segm_perm[0:i_to] + [rem_elem] + segm_perm[i_to:]

                permutation = []
                for segm_ix in segm_perm:
                    utt_ixs = sorted(getKeysByValue(segment_ixs, segm_ix))
                    permutation = permutation + utt_ixs

                if permutation not in permutations:
                    permutations.append(permutation)
                    segment_permutations.append(segm_perm)
                    break

        return permutations, segment_permutations

    def swda_utterance_sampling(self, speaker_ixs, amount):
        segm_ixs = self.speaker_segment_ixs(speaker_ixs)
        segments = list(set(segm_ixs.values()))

        permutations = []

        for i in range(amount):
            (sentence, act, swda_name, ix) = self.draw_rand_sent()
            insert_ix = random.choice(segments)
            permutations.append((sentence, act, swda_name, ix, insert_ix))

        return permutations

    def convert_dset(self, amounts):
        # create distinct train/validation/test files. they'll correspond to the created
        # splits from the constructor
        train_output_file = os.path.join(self.data_dir, 'train', 'coherency_dset_{}.txt'.format(self.task))
        val_output_file = os.path.join(self.data_dir, 'validation', 'coherency_dset_{}.txt'.format(self.task))
        test_output_file = os.path.join(self.data_dir, 'test', 'coherency_dset_{}.txt'.format(self.task))
        if not os.path.exists(os.path.join(self.data_dir, 'train')):
            os.makedirs(os.path.join(self.data_dir, 'train'))
        if not os.path.exists(os.path.join(self.data_dir, 'validation')):
            os.makedirs(os.path.join(self.data_dir, 'validation'))
        if not os.path.exists(os.path.join(self.data_dir, 'test')):
            os.makedirs(os.path.join(self.data_dir, 'test'))

        trainfile = open(train_output_file, 'w')
        valfile = open(val_output_file, 'w')
        testfile = open(test_output_file, 'w')

        shuffled_path = os.path.join(self.data_dir, "shuffled_{}".format(self.task))
        if not os.path.isdir(shuffled_path):
            os.mkdir(shuffled_path)

        for i,trans in enumerate(tqdm(self.corpus.iter_transcripts(display_progress=False), total=1155)):
            utterances = []
            acts = []
            speaker_ixs = []
            prev_act = "%"
            for utt in trans.utterances:
                sentence = re.sub(r"([+/\}\[\]]|\{\w)", "",
                                utt.text)
                sentence = self.word2id(self.tokenizer(sentence))
                utterances.append(sentence)
                act = utt.damsl_act_tag()
                if act == None: act = "%"
                if act == "+": act = prev_act
                acts.append(self.da2num[act])
                prev_act = act
                if "A" in utt.caller:
                    speaker_ixs.append(0)
                else:
                    speaker_ixs.append(1)

            if self.task == 'up':
                permuted_ixs , segment_perms = self.swda_permute(utterances, amounts, speaker_ixs)
            elif self.task == 'us':
                permuted_ixs = self.swda_utterance_sampling(speaker_ixs, amounts)
            elif self.task == 'hup':
                permuted_ixs , segment_perms = self.swda_half_perturb(amounts, speaker_ixs)
            elif self.task == 'ui':
                permuted_ixs, segment_perms = self.swda_utterance_insertion(speaker_ixs, amounts)

            swda_fname = os.path.split(trans.swda_filename)[1]
            shuffle_file = os.path.join(shuffled_path, swda_fname) # [:-4]
            with open(shuffle_file, "w") as f:
                csv_writer = csv.writer(f)
                if self.task == 'us':
                    for perm in permuted_ixs:
                        (utt, da, name, ix, insert_ix) = perm
                        row = [name, ix,insert_ix]
                        csv_writer.writerow(row)
                else:
                    for perm in segment_perms:
                        csv_writer.writerow(perm)

            if self.task == 'us':
                for p in permuted_ixs:
                    a = " ".join([str(x) for x in acts])
                    u = str(utterances)
                    insert_sent, insert_da, name, ix, insert_ix = p
                    insert_da = self.da2num[insert_da]
                    p_a = deepcopy(acts)
                    p_a[insert_ix] = insert_da
                    pa = " ".join([str(x) for x in p_a])
                    p_u = deepcopy(utterances)
                    p_u[insert_ix] = insert_sent

                    if i in self.train_ixs:
                        trainfile.write("{}|{}|{}|{}|{}\n".format("0",a,u,pa,p_u))
                        trainfile.write("{}|{}|{}|{}|{}\n".format("1",pa,p_u,a,u))
                    if i in self.val_ixs:
                        valfile.write("{}|{}|{}|{}|{}\n".format("0",a,u,pa,p_u))
                        valfile.write("{}|{}|{}|{}|{}\n".format("1",pa,p_u,a,u))
                    if i in self.test_ixs:
                        testfile.write("{}|{}|{}|{}|{}\n".format("0",a,u,pa,p_u))
                        testfile.write("{}|{}|{}|{}|{}\n".format("1",pa,p_u,a,u))

            else:
                for p in permuted_ixs:
                    a = " ".join([str(x) for x in acts])
                    u = str(utterances)
                    pa = [acts[i] for i in p]
                    p_a = " ".join([str(x) for x in pa])
                    pu = [utterances[i] for i in p]
                    p_u = str(pu)

                    if i in self.train_ixs:
                        trainfile.write("{}|{}|{}|{}|{}\n".format("0",a,u,p_a,p_u))
                        trainfile.write("{}|{}|{}|{}|{}\n".format("1",p_a,p_u,a,u))
                    if i in self.val_ixs:
                        valfile.write("{}|{}|{}|{}|{}\n".format("0",a,u,p_a,p_u))
                        valfile.write("{}|{}|{}|{}|{}\n".format("1",p_a,p_u,a,u))
                    if i in self.test_ixs:
                        testfile.write("{}|{}|{}|{}|{}\n".format("0",a,u,p_a,p_u))
                        testfile.write("{}|{}|{}|{}|{}\n".format("1",p_a,p_u,a,u))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir",
                        required=True,
                        type=str,
                        help="""The input directory where the files of the corpus
                        are located. """)
    parser.add_argument("--corpus",
                        required=True,
                        type=str,
                        help="""the name of the corpus to use, currently either 'DailyDialog' or 'Switchboard' """)
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--amount',
                        type=int,
                        default=20,
                        help="random seed for initialization")
    parser.add_argument('--word2id',
                        action='store_true',
                        help= "convert the words to ids")
    parser.add_argument('--task',
                        required=True,
                        type=str,
                        default="up",
                        help="""for which task the dataset should be created.
                                alternatives: up (utterance permutation)
                                              us (utterance sampling)
                                              hup (half utterance petrurbation)
                                              ui (utterance insertion, nothing directly added!)""")

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.word2id:
        f = open(os.path.join(args.datadir, "itos.txt"), "r")
        word2id_dict = dict()
        for i, word in enumerate(f):
            word2id_dict[word[:-1].lower()] = i

        word2id = lambda x: [word2id_dict[y] for y in x] # don't convert words to ids (yet). It gets done in the glove wrapper of mtl_coherence.py
    else:
        word2id = lambda x: x

    tokenizer = word_tokenize
    if args.corpus == 'DailyDialog':
        converter = DailyDialogConverter(args.datadir, tokenizer, word2id, task=args.task)
        converter.create_act_utt()
    elif args.corpus == 'Switchboard':
        converter = SwitchboardConverter(args.datadir, tokenizer, word2id, args.task, args.seed)
        converter.create_vocab()

    converter.convert_dset(amounts=args.amount)

def getKeysByValue(dictOfElements, valueToFind):
    listOfKeys = list()
    for item in dictOfElements.items():
        if item[1] == valueToFind:
            listOfKeys.append(item[0])
    return  listOfKeys

def switchboard_da_mapping():
    mapping_dict = dict({
                "sd": 1,
                "b": 2,
                "sv": 3,
                "aa": 4,
                "%-": 5,
                "ba": 6,
                "qy": 7,
                "x": 8,
                "ny": 9,
                "fc": 10,
                "%": 11,
                "qw": 12,
                "nn": 13,
                "bk": 14,
                "h": 15,
                "qy^d": 16,
                "o": 17,
                "bh": 18,
                "^q": 19,
                "bf": 20,
                "na": 21,
                "ny^e": 22,
                "ad": 23,
                "^2": 24,
                "b^m": 25,
                "qo": 26,
                "qh": 27,
                "^h": 28,
                "ar": 29,
                "ng": 30,
                "nn^e": 31,
                "br": 32,
                "no": 33,
                "fp": 34,
                "qrr": 35,
                "arp": 36,
                "nd": 37,
                "t3": 38,
                "oo": 39,
                "co": 40,
                "cc": 41,
                "t1": 42,
                "bd": 43,
                "aap": 44,
                "am": 45,
                "^g": 46,
                "qw^d": 47,
                "fa": 48,
                "ft":49 
            })
    d = defaultdict(lambda: 11)
    for (k, v) in mapping_dict.items():
        d[k] = v
    return d

if __name__ == "__main__":
    main()
