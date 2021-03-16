import torch
from basic.alphabet import Alphabet

class Data:
    def __init__(self,args):
        #### Global Config ####

        self.tagScheme = "BIO"  

        #### Parameters ####
        self.word_alphabet = Alphabet('word', True)

        self.asp_label_alphabet = Alphabet('aspect label', True)

        self.opi_label_alphabet = Alphabet('opinion label', True)

        self.polar_alphabet = Alphabet('relation', True)

        self.char_alphabet = Alphabet('char', True)

        self.pretrain_char_embedding = None

class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self,
                 tokens,
                 token_ids,
                 token_mask,
                 chars,
                 char_ids,
                 char_mask,
                 charLength,
                 tokenLength,
                 asp_labels,
                 opi_labels,
                 asp_label_ids,
                 opi_label_ids,
                 polaritys,
                 gold_relations):
        self.tokens = tokens #
        self.token_ids = token_ids
        self.token_mask = token_mask
        self.tokenLength = tokenLength
        self.asp_labels = asp_labels #
        self.asp_label_ids = asp_label_ids
        self.opi_labels = opi_labels  #
        self.opi_label_ids = opi_label_ids
        self.polaritys = polaritys
        self.gold_relations = gold_relations #
        self.chars = chars
        self.char_ids = char_ids
        self.char_mask = char_mask
        self.charLength = charLength


def make_data(train_features, ifgpu=False):
    all_input_ids = torch.tensor([f.token_ids for f in train_features], dtype=torch.long)
    batchSize = all_input_ids.size(0)
    all_input_mask = torch.tensor([f.token_mask for f in train_features], dtype=torch.long)
    input_length = torch.tensor([f.tokenLength for f in train_features], dtype=torch.long).view(batchSize)
    # print(input_length)
    all_char_ids = torch.tensor([f.char_ids for f in train_features], dtype=torch.long)
    char_length = torch.tensor([f.charLength for f in train_features], dtype=torch.long)
    char_mask = torch.tensor([f.char_mask for f in train_features], dtype=torch.long)

    #test use same for asp and opi
    asp_labels = torch.tensor([f.asp_label_ids for f in train_features], dtype=torch.long)
    opi_labels = torch.stack([torch.tensor(f.opi_label_ids.todense(),dtype=torch.long) for f in train_features])
    polaritys = torch.tensor([f.polaritys for f in train_features],dtype=torch.long)

    # cut by batch max seq len
    seqLen = torch.max(input_length)
    charLen = torch.max(char_length)
    all_input_ids = all_input_ids[:, :seqLen]
    all_input_mask = all_input_mask[:, :seqLen]

    char_length = char_length[:, :seqLen]
    char_length = char_length + char_length.eq(0).long()
    # print(char_length)
    all_char_ids = all_char_ids[:, :seqLen, :charLen]
    char_mask = char_mask[:, :seqLen, :charLen]

    asp_labels = asp_labels[:, :seqLen]
    opi_labels = opi_labels[:, :seqLen,:seqLen]
    polaritys = polaritys[:,:seqLen]

    # permute by length desc
    input_length, word_perm_idx = input_length.sort(0, descending=True)
    all_input_ids = all_input_ids[word_perm_idx]
    all_input_mask = all_input_mask[word_perm_idx]
    all_char_ids = all_char_ids[word_perm_idx]
    char_length = char_length[word_perm_idx]
    char_mask = char_mask[word_perm_idx]
    polaritys = polaritys[word_perm_idx]
    asp_labels = asp_labels[word_perm_idx]
    opi_labels = opi_labels[word_perm_idx]

    all_char_ids = all_char_ids.view(int(batchSize) * int(seqLen), -1)
    char_mask = char_mask.view(int(batchSize) * int(seqLen), -1)
    char_length = char_length.view(int(batchSize) * int(seqLen), )
    char_length, char_perm_idx = char_length.sort(0, descending=True)
    all_char_ids = all_char_ids[char_perm_idx]
    char_mask = char_mask[char_perm_idx]

    _, char_recover = char_perm_idx.sort(0, descending=False)
    _, input_recover = word_perm_idx.sort(0, descending=False)

    if ifgpu:
        all_input_ids = all_input_ids.cuda()
        all_input_mask = all_input_mask.cuda()
        input_length = input_length.cuda()
        input_recover = input_recover.cuda()
        all_char_ids = all_char_ids.cuda()
        char_length = char_length.cuda()
        char_recover = char_recover.cuda()
        char_mask = char_mask.cuda()

        asp_labels = asp_labels.cuda()
        opi_labels = opi_labels.cuda()
        polaritys = polaritys.cuda()

    return all_input_ids, input_length, input_recover, all_input_mask, all_char_ids, char_length, char_recover,\
           char_mask, asp_labels, opi_labels, polaritys
