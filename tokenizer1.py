# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-02-16T16:21:42.786995Z","iopub.execute_input":"2022-02-16T16:21:42.787613Z","iopub.status.idle":"2022-02-16T16:21:46.206055Z","shell.execute_reply.started":"2022-02-16T16:21:42.787484Z","shell.execute_reply":"2022-02-16T16:21:46.204858Z"}}
import pandas as pd
import numpy as np
import ast
import json
import unicodedata
from transformers import BertTokenizer
from tqdm import tqdm
import tensorflow as tf
import six


# %% [code] {"execution":{"iopub.status.busy":"2022-02-15T13:01:54.696839Z","iopub.execute_input":"2022-02-15T13:01:54.697367Z","iopub.status.idle":"2022-02-15T13:01:54.93546Z","shell.execute_reply.started":"2022-02-15T13:01:54.697309Z","shell.execute_reply":"2022-02-15T13:01:54.934578Z"},"jupyter":{"outputs_hidden":false}}
# df= pd.read_csv(path_to_text)
# df= df['text']

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-02-16T16:21:46.207993Z","iopub.execute_input":"2022-02-16T16:21:46.208244Z","iopub.status.idle":"2022-02-16T16:21:46.226279Z","shell.execute_reply.started":"2022-02-16T16:21:46.208213Z","shell.execute_reply":"2022-02-16T16:21:46.224998Z"}}
##Citation https://github.com/google-research/bert/blob/master/tokenization.py
def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    # All the whitespace characters are removed here
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text.decode("utf-8", "ignore")
        elif isinstance(text, unicode):
            return text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


class WordpieceTokenizer(object):
    """Runs WordPiece tokenziation."""

    def __init__(self, vocab, unk_token="[UNK]", max_input_chars_per_word=200):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
        """Tokenizes a piece of text into its word pieces.

        This uses a greedy longest-match-first algorithm to perform tokenization
        using the given vocabulary.

        For example:
          input = "unaffable"
          output = ["un", "##aff", "##able"]

        Args:
          text: A single token or whitespace separated tokens. This should have
            already been passed through `BasicTokenizer.

        Returns:
          A list of wordpiece tokens.
        """
        text = convert_to_unicode(text)
        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens


# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-02-16T17:11:19.472376Z","iopub.execute_input":"2022-02-16T17:11:19.472750Z","iopub.status.idle":"2022-02-16T17:11:19.529401Z","shell.execute_reply.started":"2022-02-16T17:11:19.472702Z","shell.execute_reply":"2022-02-16T17:11:19.528705Z"}}
class creating_pretraining_data:

    def __init__(self, path_to_text, path_to_vocab, path_to_output, masking_per=0.15, sentence_len=512):

        '''
        path_to_text= path to the text data
        path_to_vocab= path to the vocabulary file
        '''
        self.path_to_text = path_to_text
        self.path_to_vocab = path_to_vocab
        self.per = masking_per
        self.sentence_len = sentence_len
        self.output = path_to_output

        # Reading the vocab.txt as dictionary
        #         f = open(self.path_to_vocab)
        #         vocab_txt = f.readline()
        #         f.close()
        #         vocab_dict= json.loads(vocab_txt)
        #         self.vocab= vocab_dict['model']['vocab']

        with open(self.path_to_vocab) as file_in:
            self.vocab = {}
            count = 0
            for line in tqdm(file_in):
                self.vocab[line.replace('\n', '')] = count
                count += 1

        # wordpiece tokenizer
        self.wordpiece = WordpieceTokenizer(self.vocab)

    def token_to_vocab(self):
        return {self.vocab[key]: key for key in list(self.vocab.keys())}

    # Reading the text data line by line
    def reading_file(self, df):
        input_data = []
        self.token_to_vocab()

        # Creating an writer which will write examples into TFRECORDS file

        writer = tf.io.TFRecordWriter(self.output)
        for text in tqdm(df):
            #             print('the text is',text)
            #             print('******************************************************************')
            input_data = self.create_data(text)

            if type(input_data) == dict:
                self.parse_single_data_and_write(input_data, writer)

            else:
                for inp in input_data:
                    self.parse_single_data_and_write(inp, writer)
        writer.close()
        return

    def _bytes_feature(self, value):
        """Returns a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))):  # if value ist tensor
            value = value.numpy()  # get value of tensor
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def serialize_array(self, array):
        array = tf.io.serialize_tensor(array)
        return array

    # Writting data to tfrecord file
    def parse_single_data_and_write(self, inputs, writer):
        """
        input_data:dictionary which contains "input_token", "position" where the tokens has been masked
        and "position value" the actual value of the masked token
        """
        #         print(inputs['input_token'])
        #         print(inputs['positions'])
        #         print(inputs['positions_value'])

        # Creating an feature
        data = {
            'input_token': self._bytes_feature(tf.constant(self.serialize_array(inputs['input_token']))),
            'positions': self._bytes_feature(tf.constant(self.serialize_array(inputs['positions']))),
            'position_value': self._bytes_feature(tf.constant(self.serialize_array(inputs['positions_value']))),
            'attention_mask': self._bytes_feature(tf.constant(self.serialize_array(inputs['attention_mask'])))
        }

        # create an Example, wrapping the single features
        out = tf.train.Example(features=tf.train.Features(feature=data))
        # Writing the example in .tfrecode file
        writer.write(out.SerializeToString())

        return

        # Selecting the text such that it ends in matras ie'|

    def finding_matra(self, tok):

        # if the token is larger than 512 words than truncate it and identify the purnabiram "|"
        if len(tok) > self.sentence_len - 2:
            tok = tok[:self.sentence_len - 2]

        # Finding the position of all purnabiram "|" in the text
        temp = [i for i, j in enumerate(tok) if j == self.vocab['।']]

        # there is no purnabiram return the
        if len(temp) == 0:
            return len(tok) - 1, False
        else:
            return [i for i, j in enumerate(tok) if j == self.vocab['।']][-1] + 1, True

    # It takes single text
    def create_data(self, text):
        """
        input:
        text: It takes a single text data from the dataframe

        Returns:
        input={"position","input_token","position_value"}
        """
        input_token = []
        list_content = text.split(' ')
        self.sub_work_position = []

        # converting word into tokens
        for word in list_content:

            try:
                input_token.append(self.vocab[word])

                # If a word cannot be separated into its subword then mask its position as 0
                self.sub_work_position.append(0)


            # Strip the text according to purnunciation
            except KeyError:
                # Breaking the text into phonetics
                phonetics_tokens = [self.vocab[phonetic] for phonetic in self.wordpiece.tokenize(word)]
                input_token.extend(phonetics_tokens)

                # Finding the position of the tokens which are subwords
                for i in range(1, len(self.wordpiece.tokenize(word)) + 1):
                    self.sub_work_position.append(i)

        #         print('The total subwords position',len(self.sub_work_position))
        #         print('The total subwords position',len(input_token))

        if len(input_token) < self.sentence_len - 2:  # Without counting [cls] and [sep] token the size will be 510
            input_value = self.single_data(input_token, self.sub_work_position, padding=True)
            return input_value

        else:
            inputs = []
            start = 0
            end, purna_identifier = self.finding_matra(input_token[start:])
            end = end + start

            # Slect portion so that the last word is the '|'
            portion = input_token[start:end]

            while True:
                # As [CLS] and [SEP] token is not added
                if len(portion) == self.sentence_len - 2:
                    input_value = self.single_data(portion, self.sub_work_position[start:end], padding=False)
                else:
                    input_value = self.single_data(portion, self.sub_work_position[start:end], padding=True)

                inputs.append(input_value)

                # Breaking out if the remaining words is less than 30
                if len(input_token) - end < 30:
                    break

                start = end
                end, purna_identifier = self.finding_matra(input_token[start:])
                end = end + start

                portion = input_token[start:end]

                # If there is no purnibiram in the text don't read further
                if purna_identifier == False:
                    break
            return inputs

    def single_data(self, input_token, subword_pos, padding=True):
        """
        input_token: Tokens of the text
        subword_pos: Identifier which identify whether a word is an subword or not
        """

        # Calculating the number of text to be masked
        number_to_masked = int(len(input_token) * self.per)

        #             print('The value of self.per ',self.per)
        # Masking the data in between
        input_value = self.create_mask(input_token, subword_pos, number_to_masked)

        # Adding the [cls] at beginning and [sep] at end
        input_token.insert(0, self.vocab['[CLS]'])
        input_token.insert(len(input_token), self.vocab['[SEP]'])

        if padding == True:

            # Padding upto 512 value
            attention_mask = list(np.ones(len(input_value['input_token']), dtype=int))
            attetion_mask = np.array(
                attention_mask.extend(np.zeros(self.sentence_len - len(input_value['input_token']), dtype=int)))
            input_value['input_token'].extend(np.zeros(self.sentence_len - len(input_value['input_token']), dtype=int))
            input_value['attention_mask'] = attention_mask

        else:
            input_value['attention_mask'] = np.ones(self.sentence_len, dtype=int)

        return input_value

    def create_mask(self, input_token, subword_pos, number_to_masked):

        """
        Inputs:
        input_token: The tokens that needed to be masked
        number_to_masked: int defining the number of tokens which are going to be masked in input_token
        """
        position = []
        position_value = []
        id_ = 0

        #         print('The input_token',len(input_token))
        #         print('The subword_pos',len(subword_pos))
        #         print('THe input_token',input_token)
        #         print('length',len(input_token))
        #         print('******************************************\n')
        #         print('we are inside the mask function')
        while len(position) < number_to_masked + 1:

            if np.random.uniform(low=0.0, high=1.0, size=None) <= self.per:
                #
                # IF the word is not the subword
                if subword_pos[id_] == 0:
                    input_token, position, position_value = self.fixer(input_token, position, position_value, id_)

                else:
                    # Backward traking
                    tracker = id_
                    while subword_pos[tracker] != 0:
                        input_token, position, position_value = self.fixer(input_token, position, position_value,
                                                                           tracker)

                        # If the first token is a subword then we don;t have to track backward
                        if id_ == 0:
                            break
                        tracker = tracker - 1

                    # Forward traking
                    # If the last token is a subword then we don;t have to forward backward we can go out the loop

                    num_of_formward_Jump = 0
                    tracker = id_ + 1

                    # If tracker exceed the total number of token break
                    if tracker >= len(subword_pos):
                        break

                    while subword_pos[tracker] != 0:
                        input_token, position, position_value = self.fixer(input_token, position, position_value,
                                                                           tracker)
                        tracker = tracker + 1
                        num_of_formward_Jump += 1

                    # We should not go through the subword that are already masked
                    id_ = id_ + num_of_formward_Jump

            id_ += 1
            if id_ >= len(subword_pos):
                #                 print('The value of the id',id_)
                #                 print('The value of the subword_pos',len(subword_pos))
                break

        #         print('The value of the position',len(position))
        #         print('The value of the number to be masked',number_to_masked)
        #         print('**********************************************************')
        position = [pos + 1 for pos in position]
        position.sort()

        input_value = {'input_token': input_token, 'positions': position, 'positions_value': position_value}
        return input_value

    def fixer(self, input_token, position, position_value, id_):
        position.append(id_)
        position_value.append(input_token[id_])
        input_token[id_] = self.vocab['[MASK]']
        return input_token, position, position_value

    def encoder(self, text):

        print('The nepali text\n', text)
        return self.create_data(text)

    def decoder(self, input_token):
        token_to_vocab = self.token_to_vocab()
        text = [token_to_vocab[input] for input in input_token]
        text = ' '.join([str(elem) for elem in text])
        return text


# %% [code] {"execution":{"iopub.status.busy":"2022-02-16T16:43:49.434612Z","iopub.execute_input":"2022-02-16T16:43:49.434932Z","iopub.status.idle":"2022-02-16T16:43:49.782999Z","shell.execute_reply.started":"2022-02-16T16:43:49.434898Z","shell.execute_reply":"2022-02-16T16:43:49.781487Z"},"jupyter":{"outputs_hidden":true}}
path_to_vocab = '../input/nepalivocab/nepali-vocab.txt'
path_to_text = '../input/final-bert-corpus/Final_Data_Corpus.csv'
path_to_output = './nepali.tfrecords'
percentage = 0.15
sen_len = 512

df = pd.read_csv(path_to_text)
preprocessor = creating_pretraining_data(path_to_text, path_to_vocab, path_to_output, percentage, sen_len)
preprocessor.reading_file(df)

# %% [code] {"jupyter":{"outputs_hidden":true},"execution":{"iopub.status.busy":"2022-02-16T17:11:24.858712Z","iopub.execute_input":"2022-02-16T17:11:24.859463Z","iopub.status.idle":"2022-02-16T17:11:24.955618Z","shell.execute_reply.started":"2022-02-16T17:11:24.859423Z","shell.execute_reply":"2022-02-16T17:11:24.954400Z"}}
# # path_to_vocab= '../input/nepalivocab/nepali-vocab.txt'
# # path_to_text= '../input/kantipur-data/final kantipur.csv'
# # path_to_output= './nepali.tfrecords'
# # percentage= 0.15
# # sen_len= 100

# # preprocessor= creating_pretraining_data(path_to_text,path_to_vocab,path_to_output,percentage,sen_len)
# # # preprocessor.reading_file(df)

# nepali_text= df['text'].loc[42]

# preprocessor= creating_pretraining_data(path_to_text,path_to_vocab,path_to_output,percentage,512)
# input_data= preprocessor.encoder(nepali_text)


# # #Displaying the preprocessed data
# if type(input_data)!=list:
#     print('\n The tokenized data is:\n',preprocessor.decoder(input_data['input_token']))
#     print('\nThe tokens for the text are:\n', input_data['input_token'])
#     print('\nThe position where the masking has been done:\n',input_data['positions'])
#     print('\nThe token value for the masked tokens are:\n',input_data['positions_value'])
#     print('\n The attention masked is:\n',input_data['attention_mask'])
# else:
#     for data in input_data:

#         print('\n The tokenized data is:\n',preprocessor.decoder(data['input_token']))
#         print('\nThe tokens for the text are:\n', data['input_token'])
#         print('\nThe position where the masking has been done:\n',data['positions'])
#         print('\nThe token value for the masked tokens are:\n',data['positions_value'])
#         print('\n The attention masked is:\n',data['attention_mask'])
#         print('****************************************************\n')

# %% [code]
