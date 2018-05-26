import os
import sys
import math
import random 
import numpy as np 
import tensorflow as tf 
import data_processor
import seq2seq_model 

def show_progress(text):
    sys.stdout.write(text)
    sys.stdout.flush()

def read_data_into_buckets(enc_path, dec_path, buckets):
    """
    対話データの語彙数（インデックス数）と、各bucketでの配列数の定義に基づき
    data_setに（bucketIdを添え字として）インデックス配列を格納して返す

    Args:
        enc_path: 要求メッセージのパス
        dec_path: 応答メッセージのパス
        buckets: バケットのサイズを格納した配列

    Returns: バケットごとにインデックス配列を格納したdata_set
    """

    data_set = [[] for _ in buckets]
    with tf.gfile.GFile(enc_path, mode='r') as ef, tf.gfile.GFile(dec_path, mode='r') as df:
        #read tweets and replies from text file.
        tweet, reply = ef.readline(), df.readline()
        counter = 0
        while tweet and reply:
            
            # プログレス
            counter += 1
            if counter % 100000 == 0:
                print('   Reading data line %d' % counter)
                sys.stdout.flush()

            # インデックスの羅列を配列化
            # 応答側は、応答の終端を示すために末尾にEOSを追加
            source_ids = [int(x) for x in tweet.split()]
            target_ids = [int(x) for x in reply.split()]
            target_ids.append(data_processor.EOS_ID)

            # 対話のインデックス数に基づき、data_setにインデックス配列を追加していく
            for bucket_id, (source_size, target_size) in enumerate(buckets):
                #Find bucket to put this conversation based on tweet and reply length
                if len(source_ids) < source_size and len(target_ids) < target_size:
                    data_set[bucket_id].append([source_ids, target_ids])
                    break
            
            # 次のデータ
            tweet, reply = ef.readline(), df.readline()

    # data_setに格納したインデックス配列数をコンソールに出力
    for bucket_id in range(len(buckets)):
        print("read_data_into_buckets: bucket={}, len={}".format(buckets[bucket_id], len(data_set[bucket_id])))
    
    return data_set

def create_or_restore_model(session, buckets, forward_only, beam_search, beam_size):
    
    # beam search is off for training
    """
    Create model and initialize or load parameters
    モデルの生成、パラメータの復元を行う。
    checkpointファイルが存在すればそれを読み込む

    Args:
        session: tfのSession
        buckets: バケット配列
        forward_only: forwardのみとするか
        beam_search: BeamSearchを使用するかのフラグ
        beam_size: BeamSearchのサイズ

    Returns:
        Seq2SeqModel
    """

    model = seq2seq_model.Seq2SeqModel(source_vocab_size=data_processor.MAX_ENC_VOCABULARY, # eg)50000 ボキャブラリのMAX
                                       target_vocab_size=data_processor.MAX_DEC_VOCABULARY, # eg)50000 ボキャブラリのMAX
                                       buckets=buckets,
                                       size=data_processor.LAYER_SIZE,          # eg)256
                                       num_layers=data_processor.NUM_LAYERS,    # eg)3
                                       max_gradient_norm=data_processor.MAX_GRADIENT_NORM,  # eg)5.0
                                       batch_size=data_processor.BATCH_SIZE,    # eg)64
                                       learning_rate=data_processor.LEARNING_RATE,  # eg)0.5
                                       learning_rate_decay_factor=data_processor.LEARNING_RATE_DECAY_FACTOR,    # eg)0.99
                                       beam_search=beam_search,
                                       attention=True,
                                       forward_only=forward_only,
                                       beam_size=beam_size  # eg)20
                                       )
    print("model initialized")

    # checkpointファイルがあれば復元
    ckpt = tf.train.get_checkpoint_state("./ckpt/.")
    checkpoint_suffix = ".index"
    if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path + checkpoint_suffix):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())

    return model

def next_random_bucket_id(buckets_scale):
    """
    バケットIDをランダムに選択して返す
    Args:
        buckets_scale: 各バケットの語彙数の割合を格納した配列
    Retuens:
        バケットID
    """

    # 0.0以上、1.0未満の疑似乱数を生成
    n = np.random.random_sample()
    
    # 乱数を超える語彙数割合を持つバケットIDを選択する
    bucket_id = min([i for i in range(len(buckets_scale)) if buckets_scale[i] > n])
    return bucket_id

def train():
    """
    トレーニング
    """

    # GPUは存在するれば、先頭一個だけ使う（visible_device_list = "0"）
    # メモリ確保の「allow_growth=True # True->必要になったら確保, False->全部」をいれるか検討
    tf_config = tf.ConfigProto(gpu_options = tf.GPUOptions(visible_device_list = "0"))
    with tf.Session(config=tf_config) as sess:

        # 対話データの語彙数（インデックス数）に基づき、インデックス配列を格納したdata_setを生成する
        show_progress("Setting up data set for each buckets...")
        train_set = read_data_into_buckets(data_processor.TRAIN_ENC_IDX_PATH,
            data_processor.TRAIN_DEC_IDX_PATH, data_processor.buckets)
        valid_set = read_data_into_buckets(data_processor.VAL_ENC_IDX_PATH,
            data_processor.VAL_DEC_IDX_PATH, data_processor.buckets)
        show_progress("done\n")

        # モデルを復元する
        show_progress("Creating model...")
        # False for train
        beam_search = False
        model = create_or_restore_model(sess, data_processor.buckets, forward_only = False,
                                beam_search = beam_search, beam_size = data_processor.beam_size)
        show_progress("done\n")

        # list of # of data in ith bucket
        # 各バケットごとのトレーニングデータの語彙数、およびトータル語彙数を取得
        train_bucket_sizes = [len(train_set[b]) for b in range(len(data_processor.buckets))]
        train_total_size = float(sum(train_bucket_sizes))

        # Originally from https://github.com/1228337123/tensorflow-seq2seq-chatbot
        # This is for choosing randomly bucket based on distribution
        # トレーニングでバケットをランダムに選択する為に、各バケットの語彙数の割合を出しておく
        train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                               for i in range(len(train_bucket_sizes))]

        # Train Loop
        show_progress("before train loop")
        steps = 0
        previous_perplexities = []
#        writer = tf.summary.FileWriter(data_processor.LOGS_DIR, sess.graph)

        # Ctrl+Cで停止
        while True:

            # バケットIDをランダムに選択
            bucket_id = next_random_bucket_id(train_buckets_scale)

            # バッチデータを生成する
            encoder_inputs, decoder_inputs, target_weights = model.get_batch(train_set, bucket_id)
            #      show_progress("Training bucket_id={0}...".format(bucket_id))

            # Train!
            # トレーニングの実施
            _, average_perplexity, _ = model.step(sess, encoder_inputs, decoder_inputs, target_weights,
                                                    bucket_id,
                                                    forward_only=False,
                                                    beam_search=beam_search)
#            _, average_perplexity, ,summary, _ = model.step(sess, encoder_inputs, decoder_inputs, target_weights,
#                                                           bucket_id,
#                                                           forward_only=False,
#                                                           beam_search=beam_search)

            #      show_progress("done {0}\n".format(average_perplexity))

            # 進捗出力
            steps = steps + 1
            if steps % 10 == 0:
#                writer.add_summary(summary, steps)
                show_progress(".")

            # 規定回数ごとにcheckpointの書き出し
#            if steps % 500 != 0:
            if steps % 100 != 0:
                continue

            # checkpointの書き出し
            checkpoint_path = "./ckpt/seq2seq.ckpt"
            show_progress("Saving checkpoint...")
            model.saver.save(sess, checkpoint_path, global_step=model.global_step)
            show_progress("done\n")

            # 現在の学習状態をコンソール出力
            perplexity = math.exp(average_perplexity) if average_perplexity < 300 else float('inf')
            print ("global step %d learning rate %.4f perplexity "
                   "%.2f" % (model.global_step.eval(), model.learning_rate.eval(), perplexity))

            # Decrease learning rate if no improvement was seen over last 3 times.
            # 過去3回で改善が見られなかった場合、学習率を下げる
            if len(previous_perplexities) > 2 and perplexity > max(previous_perplexities[-3:]):
                sess.run(model.learning_rate_decay_op)
            previous_perplexities.append(perplexity)

            # 検証
            for bucket_id in range(len(data_processor.buckets)):
                if len(valid_set[bucket_id]) == 0:
                    print("  eval: empty bucket %d" % bucket_id)
                    continue

                # バッチデータの生成
                encoder_inputs, decoder_inputs, target_weights = model.get_batch(valid_set, bucket_id)

                # 検証の実行
                _, average_perplexity, _ = model.step(sess, encoder_inputs, decoder_inputs, target_weights,
                    bucket_id, True, beam_search=beam_search)
#                writer.add_summary(valid_summary, steps)

                # 検証結果の出力
                eval_ppx = math.exp(average_perplexity) if average_perplexity < 300 else float('inf')
                print("  eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))

if __name__ == "__main__":
    print("Tensorflow version = " + tf.__version__)
    train()
    