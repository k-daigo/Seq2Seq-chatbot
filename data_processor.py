import re
import sys
from sys import platform
import tensorflow as tf

#For Japanese tokenizer
import MeCab

is_fast_build = False
beam_search = True
beam_size = 20

DATA_DIR = "data"

if is_fast_build:
    MAX_ENC_VOCABULARY = 5
    NUM_LAYERS = 2
    LAYER_SIZE = 2
    BATCH_SIZE = 2
    buckets = [(5, 10), (8, 13)]
else:
    MAX_ENC_VOCABULARY = 50000
    NUM_LAYERS = 3
    LAYER_SIZE = 256
    BATCH_SIZE = 64
    buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]

MAX_DEC_VOCABULARY = MAX_ENC_VOCABULARY

# 学習率
LEARNING_RATE = 0.5
# 学習が芳しくないときの学習率を下げる。その際の値。
LEARNING_RATE_DECAY_FACTOR = 0.99
# Clip gradients to this norm.
MAX_GRADIENT_NORM = 5.0

#path list
SOURCE_PATH = "data/source.txt"
TARGET_PATH = "data/target.txt"

TRAIN_ENC_PATH = "generated/011_train_enc.txt"
VALIDATION_ENC_PATH = "generated/021_validation_enc.txt"

TRAIN_DEC_PATH = "generated/012_train_dec.txt"
VALIDATION_DEC_PATH = "generated/022_validation_dec.txt"

VOCAB_ENC_PATH = "generated/031_vocab_enc.txt"
VOCAB_DEC_PATH = "generated/032_vocab_dec.txt"

TRAIN_ENC_IDX_PATH = "generated/041_train_enc_idx.txt"
TRAIN_DEC_IDX_PATH = "generated/042_train_dec_idx.txt"
VAL_ENC_IDX_PATH = "generated/051_validation_enc_idx.txt"
VAL_DEC_IDX_PATH = "generated/052_validation_dec_idx.txt"
LOGS_DIR = 'logs'
DIGIT_RE = re.compile(r"\d")
_WORD_SPLIT = re.compile("([.,!/?\":;)(])")

_PAD = "_PAD"
_GO = "_GO"
_EOS = "_EOS"
_UNK = "_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

# MeCabでわかち書きする
tagger = MeCab.Tagger("-Owakati")

def japanese_tokenizer(sentence):
    """
    日本語文章のトークナイザ。
    taggerで定義しているTagger?で分かち書きを行う。

    Args:
        sentence: 解析する文字列
    
    Returns:
    
    """

    #sentenceのtypeはstr??
    assert type(sentence) is str

    # わかち書きして返す
    result = tagger.parse(sentence)
    return result.split()

def num_lines(file):
    """
    指定ファイルの行数を返す

    Args:
        file: target file
    
    Returns:
        of lines in file
    """

    return sum(1 for _ in open(file))

def create_train_validation(filepath, train_path, validation_path, train_ratio = 0.9):
    """
    指定ファイルをレシオで分割して、トレーニング用のファイルと検証用のファイルに分けて書き込む

    Args:
      filepath: source file path
      train_path: path to write train data
      validation_path: path to write validation data
      train_ration = train data ratio

      returns None
    """

    # レシオを元に、トレーニング対象とする行数を算出する
    nb_lines = num_lines(filepath)
    nb_train = int(nb_lines * train_ratio)

    # 指定ファイルを分割してトレーニング用と、検証用のファイルに書き込む
    counter = 0
    with tf.gfile.GFile(filepath, "r") as f, tf.gfile.GFile(train_path, "w") as trf, tf.gfile.GFile(validation_path, "w") as vlf:
        for line in f:
            if counter < nb_train:
                trf.write(line)
            else:
                vlf.write(line)

            counter = counter + 1
        

def create_vocabulary(filepath, vocabulary_path, max_vocabulary_size, tokenizer = japanese_tokenizer):
    """
    単語ファイル（vocabulary_path）に分かち書きで分割した単語を出力する

    Args:
        filepath: ソースファイルのパス
        vocabulary_path: ボキャブラリファイルのパス
        max_vocabulary_size: ボキャブラリファイルに書き出す最大件数
        tokenizer: トークナイザ
    returns: None
    """

    if tf.gfile.Exists(vocabulary_path):
        print("Found vocabulary file")
        return

    with tf.gfile.GFile(filepath, "r") as f:
        counter = 0
        vocab = {} #word:word_freq

        for line in f:
            counter += 1

            # 分かち書き結果を得る
            words = tokenizer(line)

            # 5000行に１回の進捗表示
            if counter % 5000 == 0:
                sys.stdout.write(".")
                sys.stdout.flush()

            for word in words:
                
                # 数値を「0」に置換
                word = re.sub(DIGIT_RE, "0", word)

                # 単語の数をvocabに入れる
                if word in vocab:
                    vocab[word] += 1
                else:
                    vocab[word] = 1
        
        # _PAD, _GO, _EOS, _UNK, 単語1, 単語2, ... というボキャブラリ配列を作る
        # 配列数の上限に達したら上限件数で配列を切り捨てる
        vocab_list = _START_VOCAB + sorted(vocab, key = vocab.get, reverse = True)
        if len(vocab_list) > max_vocabulary_size:
            vocab_list = vocab_list[:max_vocabulary_size]

        # 配列の内容をvocabulary_pathに書き出す
        with tf.gfile.GFile(vocabulary_path, "w") as vocab_file:
            for w in vocab_list:
                vocab_file.write(w + "\n")

def basic_tokenizer(sentence):
    words = []
    for space_separated_fragment in sentence.strip().split():
        words.extend(re.split(_WORD_SPLIT, space_separated_fragment))
    return [w for w in words in w]

def sentence_to_token_ids(sentence, vocabulary, tokenizer = japanese_tokenizer, normalize_digits = True):
    """
        指定の文章について分かち書きを行い、それをボキャブラリにぶつけた結果の配列を返す

    Args:
        sentence: 文章
        vocabulary: (word, index) のDictionary
        tokenizer: トークナイザ
        normalize_digits : 
    
    Returns:
        指定の文章を分かち書きし、存在する場合はそのIndex、存在しない場合はUNK(3)を格納した配列を返す
    """

    # 分かち書きを取得
    if tokenizer:
        words = tokenizer(sentence)
    else:
        words = basic_tokenizer(sentence)
    
    # 分かち書き文章をボキャブラリにぶつけて、存在する場合はそのIndex、存在しない場合はUNK(3)を返す
    # というか、処理一緒ｗｗ
    if not normalize_digits:
        return [vocabulary.get(w, UNK_ID) for w in words]

    return [vocabulary.get(w, UNK_ID) for w in words]

def data_to_token_ids(data_path, target_path, vocabulary_path,
             tokenizer = japanese_tokenizer, normalize_digits = True):
    """
    指定のファイルに記載されている文章を分かち書きし、そのボキャブラリIndexを半角スペース区切りでtarget_pathに出力する

    Args:
        data_path: 
        target_path: 
        vocabulary_path: 
        tokenizer: 
        normalize_digits:

    Returns: 
    """

    if tf.gfile.Exists(target_path):
        return

    print("Tokenizing data in %s" % data_path)

    # ボキャブラリファイルを元に(word, index) のdictを取得
    vocab, _ = initialize_vocabulary(vocabulary_path)

    with tf.gfile.GFile(data_path, "rb") as data_file:
        with tf.gfile.GFile(target_path, "wb") as tokens_file:
            counter = 0
            for line in data_file:
                counter += 1

                # 進捗を出力
                if counter % 1000000 == 0:
                    print("   tokenizing line %d" % counter)
                
                # バイナリで読んでいるのでUTF-8で読み直し？？
                line = line.decode('utf-8')

                # 指定の文章について分かち書きを行い、それをボキャブラリにぶつけた結果のボキャブラリIndexの配列を返す
                token_ids = sentence_to_token_ids(line, vocab, tokenizer, normalize_digits)
                
                # 半角スペース区切りのボキャブラリIndexの配列をtarget_pathに出力する
                tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")
        
def initialize_vocabulary(vocabulary_path):
    """
    ボキャブラリファイルを元に(word, index) のdictを返す
    
    Args:
        vocabulary_path: ボキャブラリファイルのパス
    
    Returns:
        (word, index) のdict
    """

    if tf.gfile.Exists(vocabulary_path):
        rev_vocab = []

        # 既存のボキャブラリファイルの内容をrev_vocabに追加
        with tf.gfile.GFile(vocabulary_path, "r") as f:
            rev_vocab.extend(f.readlines())

        # 各単語の先頭、末尾の空白文字を除去
        rev_vocab = [line.strip() for line in rev_vocab]

        # rev_vocab を (word, index) でdict化
        #Dictionary of (word, idx)
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocabulary_path)


if __name__ == "__main__":
    print("Tensorflow version = " + tf.__version__)

    # ソースファイルをレシオで分割して、トレーニング用と検証用のファイルに分割して書き込む
    print("Splitting into train and validation data...")
    create_train_validation(SOURCE_PATH, TRAIN_ENC_PATH, VALIDATION_ENC_PATH)

    # ターゲットファイルをレシオで分割して、トレーニング用と検証用のファイルに分割して書き込む
    create_train_validation(TARGET_PATH, TRAIN_DEC_PATH, VALIDATION_DEC_PATH)
    print("Done")

    # 単語ファイル（vocabulary_path）に分かち書きで分割した単語を出力する
    print("Creating vocabulary files...")
    # ソースファイル分
    create_vocabulary(SOURCE_PATH, VOCAB_ENC_PATH, MAX_ENC_VOCABULARY)
    # ターゲットファイル分
    create_vocabulary(TARGET_PATH, VOCAB_DEC_PATH, MAX_DEC_VOCABULARY)
    print("Done")

    # 各ファイルに記載されている文章を分かち書きし、ボキャブラリIndexを半角スペース区切りでtarget_pathに出力する
    print("Creating sentence idx files...")
    data_to_token_ids(TRAIN_ENC_PATH, TRAIN_ENC_IDX_PATH, VOCAB_ENC_PATH)
    data_to_token_ids(TRAIN_DEC_PATH, TRAIN_DEC_IDX_PATH, VOCAB_DEC_PATH)
    data_to_token_ids(VALIDATION_ENC_PATH, VAL_ENC_IDX_PATH, VOCAB_ENC_PATH)
    data_to_token_ids(VALIDATION_DEC_PATH, VAL_DEC_IDX_PATH, VOCAB_DEC_PATH)
    print("Done")
