"""
Contributor:  Micheleen Harris
Date:  Feb. 20, 2022
Original source:  https://github.com/donglee-afar/logdeep/issues/3#issuecomment-750028771
Purpose:  Map event ids to an encoded semantics vector (specifically for loganomaly method)
         将事件 ID 映射到一个编码后的语义向量（专门用于 LogAnomaly 方法）。
Notes:
- Uses the spellpy parser project:  https://github.com/nailo2c/spellpy (need to pip install)
- Need the stop words Python file from SpaCy in project folder with this file:  https://github.com/explosion/spaCy/blob/master/spacy/lang/en/stop_words.py
- Example below are from Ubuntu system logs (normal and abnormal as deemed by user)

	•	本代码使用 spellpy 解析器项目：spellpy（需要使用 pip install 安装）。
	•	需要 SpaCy 的 停用词列表，请将以下文件放入项目文件夹中：stop_words.py。
	•	下面的示例数据来自 Ubuntu 系统日志（包括正常日志和用户认为的异常日志）。

Get "cc.en.300.vec" by (on Linux; note, the unarchived file is ~4.5 GB):
获取 FastText 预训练词向量 cc.en.300.vec（Linux 命令，解压后约 4.5GB）：

mkdir vec_models
cd vec_models
wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.vec.gz
gunzip cc.en.300.vec.gz
"""
import re
import json
import io
from tqdm import tqdm
import os
import numpy as np
import pandas as pd
from collections import OrderedDict, Counter
import math
from pprint import pprint
from datetime import datetime

from stop_words import StopWords

from spellpy import spell


class Preprocessor:

    def __init__(self):
        self.stop_words = StopWords().STOP_WORDS

    def df_transfer(self, df, event_id_map):
        year = [str(datetime.utcnow().year)] * df.shape[0]
        timestamps = list(map(lambda a, b, c, d: a + '-' + b + '-' + str(c).rstrip() + ' ' + str(d),
                              year,
                              df['Month'],
                              df['Day'],
                              df['Time']))
        df['datetime'] = pd.to_datetime(timestamps, errors='coerce')
        df.dropna(inplace=True)
        df = df[['datetime', 'EventId']]
        df['EventId'] = df['EventId'].apply(lambda e: event_id_map[e] if event_id_map.get(e) else -1)
        deeplog_df = df.set_index('datetime').resample('1min').apply(self._custom_resampler).reset_index()
        return deeplog_df

    def _custom_resampler(self, array_like):
        """Can sample however is needed"""
        return list(array_like)

    def file_generator(self, filename, df):
        with open(filename, 'w') as f:
            for event_id_list in df['EventId']:
                for event_id in event_id_list:
                    f.write(str(event_id) + ' ')
                if len(event_id_list) > 0:
                    f.write('\n')

    def normalize_text(self, text):
        """
        Normalize text to extract most salient tokens
        Ref: https://github.com/MLWorkshops/nlp-dealing-with-text-data/blob/master/Dealing-with-text-data.ipynb
        Ref: turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
        """
        # replace special characters with space and remove digits
        text = re.sub(r'\W+', ' ', text)
        text = re.sub('\d', '', text)
        # convert camel case to snake case, then replace _ with space
        text = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', text)
        text = re.sub('([a-z0-9])([A-Z])', r'\1_\2', text).lower().replace('_', ' ')
        # tokenize, removing stop words (from SpaCy)
        normalized_tokens = [t for t in text.split(' ') if t not in self.stop_words and t != '']
        return normalized_tokens

    def dump2json(self, dump_dict, target_path):
        """
        Save json and any bytes-like objects to file
        """

        class MyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, bytes):
                    return str(obj, encoding='utf-8')
                return json.JSONEncoder.default(self, obj)

        with open(target_path, 'w', encoding='utf-8') as file:
            file.write(json.dumps(dump_dict, cls=MyEncoder, indent=4))

    def create_word2idf(self, log_train, eventid2template):
        """
        Create a word to IDF dict
        TF = term frequency
        IDF = inverse document frequency
        """
        idf_matrix = list()
        for seq in log_train['EventId']:
            for event in seq:
                idf_matrix.append(eventid2template[event])
        idf_matrix = np.array(idf_matrix)

        X_counts = []
        for i in range(idf_matrix.shape[0]):
            word_counts = Counter(idf_matrix[i])
            X_counts.append(word_counts)

        X_df = pd.DataFrame(X_counts)
        X_df = X_df.fillna(0)
        events = X_df.columns
        X = X_df.values
        num_instance, num_event = X.shape
        df_vec = np.sum(X > 0, axis=0)

        # smooth idf like sklearn
        idf_vec = np.log((num_instance + 1) / (df_vec + 1)) + 1
        print(idf_vec)
        idf_matrix = X * np.tile(idf_vec, (num_instance, 1))
        X_new = idf_matrix

        word2idf = dict()
        for i, j in zip(events, idf_vec):
            word2idf[i] = j
            # smooth idf when oov
            word2idf['oov'] = (math.log((num_instance + 1) / (29 + 1)) + 1)
        return word2idf

    def create_semantic_vec(self, eventid2template, fasttext_map, word2idf):
        event2semantic_vec = dict()
        for event in eventid2template.keys():
            template = eventid2template[event]
            tem_len = len(template)
            count = dict(Counter(template))
            for word in count.keys():
                # TF
                TF = count[word] / tem_len
                # IDF
                IDF = word2idf.get(word, word2idf['oov'])
                count[word] = TF * IDF
            value_sum = sum(count.values())
            for word in count.keys():
                count[word] = count[word] / value_sum
            semantic_vec = np.zeros(300)
            for word in count.keys():
                try:
                    fasttext_weight = np.array(fasttext_map[word])
                except KeyError as ke:
                    # word not in fasttext
                    pass
                semantic_vec += count[word] * fasttext_weight
            event2semantic_vec[event] = list(semantic_vec)
        return event2semantic_vec


class FastTextProcessor:
    """Use fasttext vectors to generate map"""

    def __init__(self):
        self.template_set = set()
        self.template_fasttext_map = {}

    def create_template_set(self, result):
        print('Creating template set')
        for key in tqdm(result.keys()):
            for word in result[key]:
                self.template_set.add(word)

    def load_vectors(self, fname):
        """
        Ref: https://github.com/facebookresearch/fastText/blob/master/docs/crawl-vectors.md
        """
        fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        n, d = map(int, fin.readline().split())
        data = {}
        print('Loading vectors')
        for line in tqdm(fin):
            tokens = line.rstrip().split(' ')
            data[tokens[0]] = map(float, tokens[1:])
        return data

    def create_map(self):
        fasttext = self.load_vectors(os.path.join('vec_models', 'cc.en.300.vec'))
        print('Creating fasttext map')
        for word in tqdm(self.template_set):
            try:
                self.template_fasttext_map[word] = list(fasttext[word])
            except KeyError as ke:
                # fasttext does not have word
                pass
        return self.template_fasttext_map


if __name__ == "__main__":

    preprocessor = Preprocessor()

    ##########
    # Parser #
    ##########
    input_dir = '../data/'
    output_dir = './results_spell/'
    recreated_parse_logs = False
    # "Content" is like the log message - what we want to parse
    # the following is specific to the syslog, so match to those "columns"
    log_format = '<Month> <Day> <Time> <MachineName> <Content>'
    log_main = 'syslog'
    tau = 0.5

    parser = spell.LogParser(
        indir=input_dir,
        outdir=output_dir,
        log_format=log_format,
        logmain=log_main,
        tau=tau,
    )

    # if the we wish, we can recreate the parsed csv's
    if recreated_parse_logs:
        os.makedirs(output_dir)
        for log_name in ['syslog.1.updated',
                         'syslog.2.updated',
                         'abnormal_states.log']:
            parser.parse(log_name)

    ##################
    # Transformation #
    ##################
    # TODO:  read from object, not file i/o again
    df_train = pd.read_csv(f'{output_dir}/syslog.1.updated_structured.csv')
    df_test_normal = pd.read_csv(f'{output_dir}/syslog.2.updated_structured.csv')
    df_test_abnormal = pd.read_csv(f'{output_dir}/abnormal_states.log_structured.csv')

    print('Number of classes for training = ', df_train['EventId'].unique().shape)

    event_id_map = dict()
    for i, event_id in enumerate(df_train['EventId'].unique(), 1):
        event_id_map[event_id] = i

    # Train Set
    log_train = preprocessor.df_transfer(df_train, event_id_map)
    preprocessor.file_generator('./results_preprocessor/train', log_train)

    # Test Normal Set
    log_test_normal = preprocessor.df_transfer(df_test_normal, event_id_map)
    preprocessor.file_generator('./results_preprocessor/test_normal', log_test_normal)

    # Test Abnormal Set
    log_test_abnormal = preprocessor.df_transfer(df_test_abnormal, event_id_map)
    preprocessor.file_generator('./results_preprocessor/test_abnormal', log_test_abnormal)

    #####################
    # Event to Template #
    #####################
    eventid2template = {}
    print('Creating event IDs to templates')
    for eid in tqdm(df_train['EventId'].unique()):
        eventid2template[event_id_map[eid]] = preprocessor.normalize_text(
            df_train.loc[event_id_map[eid], 'EventTemplate'])
    preprocessor.dump2json(eventid2template, './results_preprocessor/eventid2template.json')

    ################
    # Fasttext map #
    ################
    fasttext_processor = FastTextProcessor()
    fasttext_processor.create_template_set(eventid2template)
    template_fasttext_map = fasttext_processor.create_map()
    preprocessor.dump2json(template_fasttext_map, './results_preprocessor/fasttext_map.json')

    ###############
    # Word to IDF #
    ###############
    word2idf = preprocessor.create_word2idf(log_train, eventid2template)
    preprocessor.dump2json(word2idf, './results_preprocessor/word2idf.json')

    #############################
    # Event to Semantics Vector #
    #############################
    event2semantic_vec = preprocessor.create_semantic_vec(eventid2template, template_fasttext_map, word2idf)
    preprocessor.dump2json(event2semantic_vec, './results_preprocessor/event2semantic_vec.json')
