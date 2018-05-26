#coding: utf-8

import sys 
import tensorflow as tf 
import numpy as np 
import train
import data_processor

def get_prediction(session, model, enc_vocab, rev_dec_vocab, text):
    token_ids = data_processor.sentence_to_token_ids(text, enc_vocab)

    bucket_id = min([b for b in range(len(data_processor.buckets))
                        if data_processor.buckets[b][0] > len(token_ids)])
 
    encoder_inputs, decoder_inputs, target_weights = model.get_batch({bucket_id:[(token_ids, [])]}, bucket_id)

    _, _, output_logits = model.step(session, encoder_inputs, decoder_inputs,
                                                target_weights, bucket_id, True, beam_search = False)
    
    outputs = [int(np.argmax(logit, axis = 1)) for logit in output_logits]
    if data_processor.EOS_ID in outputs:
        outputs = outputs[:outputs.index(data_processor.EOS_ID)]
    text = "".join([tf.compat.as_str(rev_dec_vocab[output]) for output in outputs])
    return text

def get_beam_search_prediction(session, model, enc_vocab, rev_dec_vocab, text):
    """"
    ビームサーチで予測を実行する
    Args:
        session: Session
        model: モデル
        enc_vocab: エンコード側ボキャブラリ（K:word, V:index）
        rev_dec_vocab: デコード側ボキャブラリ（K:word, V:index）
        text: 入力の文字列
    Returns:
        予測結果。
        data_processor.beam_sizeの（重複を除いた）件数で予測結果を返す
    """

    # 入力文字列が、（バケットで定義されている）エンコード側の最大「文字数」を超えている場合、最大で切る
#    max_len = data_processor.buckets[-1][0]
#    target_text = text
#    if len(text) > max_len:
#        target_text = text[:max_len]

    # 指定の文章について分かち書きを行い、それをボキャブラリにぶつけた結果の配列を取得する
#    token_ids = data_processor.sentence_to_token_ids(target_text, enc_vocab)

    # Add! max_lenでの末尾切りは語彙で分割してからが正しい？
    # 指定の文章について分かち書きを行い、それをボキャブラリにぶつけた結果の配列を取得する 
    token_ids_org = data_processor.sentence_to_token_ids(text, enc_vocab)
    # 入力文字列が、（バケットで定義されている）エンコード側の最大「文字数」を超えている場合、最大で切る
    max_len = data_processor.buckets[-1][0]
    token_ids = token_ids_org
    if len(token_ids) > max_len:
        token_ids = token_ids_org[:max_len]
    # Add!

    # 配列数（語彙数）を元に最小のバケットを決定する
    target_buckets = [b for b in range(len(data_processor.buckets))
                        if data_processor.buckets[b][0] > len(token_ids)]
    if not target_buckets:
        return []
    bucket_id = min(target_buckets)

    # バッチデータを生成し、予測を実行する
    # バッチの出力はtriple (encoder_inputs, decoder_inputs, target_weights) 。
    encoder_inputs, decoder_inputs, target_weights = model.get_batch({bucket_id:[(token_ids, [])]}, bucket_id)
    # 予測の出力はtriple (No gradient norm, loss, outputs)
    path, symbol, output_logits = model.step(session, encoder_inputs, decoder_inputs,
                                                target_weights, bucket_id, True,
                                                beam_search = data_processor.beam_search)

    beam_size = data_processor.beam_size
    k = output_logits[0]

    # beam_sizeの配列をpathsを作る
    paths = []
    for kk in range(beam_size):
        paths.append([])

    curr = list(range(beam_size))
    num_steps = len(path)

    # BaemSearchの場合は、予測結果の組み換えを行う必要がある？
    # pathで出力されているindexを基に、symbolから予測indexを抜き出していく→pathsに入れる
    # TODO：入れ替え方法の理解
    for i in range(num_steps - 1, -1, -1):
        for kk in range(beam_size):
            paths[kk].append(symbol[i][curr[kk]])
            curr[kk] = path[i][curr[kk]]

    recos = set()
    ret = []
    i = 0
    for kk in range(beam_size):
        # 逆順にして
        foutputs = [int(logit) for logit in paths[kk][::-1]]

        # If there is an EOS symbol in outputs, cut them at that point.
        # EOSが含まれる場合、EOSでちょん切って、それより前を使う
        if data_processor.EOS_ID in foutputs:
            foutputs = foutputs[:foutputs.index(data_processor.EOS_ID)]

        # ボキャブラリから当該インデックスの語彙を取得してつなげる
        rec = "".join([tf.compat.as_str(rev_dec_vocab[output]) for output in foutputs])

        # beam_size分、予測を詰め込んで行く（同じ予測はいれない）
        # （先頭だけ取るが）
        if rec not in recos:
            recos.add(rec)
            ret.append(rec)
    return ret

class EasyPredictor:
    def __init__(self, session):
        self.session = session

        # モデルの生成、パラメータの復元を行う。
        # checkpointファイルが存在すればそれを読み込む
        train.show_progress("Creating Model...")
        self.model = train.create_or_restore_model(self.session, data_processor.buckets, forward_only = True,
                                                     beam_search = data_processor.beam_search,
                                                     beam_size = data_processor.beam_size)
        self.model.batch_size = 1
        train.show_progress("done\n")

        # ボキャブラリファイルを元に(word, index) のdictを生成する
        self.enc_vocab, _ = data_processor.initialize_vocabulary(data_processor.VOCAB_ENC_PATH)
        _, self.rev_dec_vocab = data_processor.initialize_vocabulary(data_processor.VOCAB_DEC_PATH)

    def predict(self, text):
        """
        予測の実行
        """
        if data_processor.beam_search:
            replies = get_beam_search_prediction(self.session, self.model, self.enc_vocab,
                                                self.rev_dec_vocab, text)
            return replies
        else:
            reply = get_prediction(self.session, self.model, self.enc_vocab, self.rev_dec_vocab, text)
            return [reply]

def predict():
    """
    入力された文字列を元に予測を実行し、応答を出力する
    """

    tf_config = tf.ConfigProto(gpu_options = tf.GPUOptions(visible_device_list = "0"))
    with tf.Session(config=tf_config) as sess:
        predictor = EasyPredictor(sess)

        # 入力の受付
        sys.stdout.write("> ")
        sys.stdout.flush()
        line = sys.stdin.readline()
        while line:

            # 予測の実行
            replies = predictor.predict(line)
            #for i, text in enumerate(replies):
                #print(i, text)

            # 第一位の予測結果を出力    
            print(replies[0])

            # 入力の受付
            print("> ", end = "")
            sys.stdout.flush()
            line = sys.stdin.readline()

if __name__ == "__main__":
    print("Tensorflow version = " + tf.__version__)
    predict()
