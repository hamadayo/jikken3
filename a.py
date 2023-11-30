import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

from tensorflow.keras import layers 
from tensorflow.keras import models # 高レベルのニューラルネットワークAPI
from IPython import display

# Set the seed value for experiment reproducibility.

seed = 42                 # 実験の再現性を確保するためにシード値が設定されています。シードを設定すると、ランダム操作がコードを実行するたびに同じ結果を生成
tf.random.set_seed(seed)  # TensorFlowのランダムシードを設定します
np.random.seed(seed)      # NumPyのランダムシードを設定します

# ミニ音声コマンドデータセットをインポートする

DATASET_PATH = 'data/mini_speech_commands' # ダウンロードおよび解凍されるデータセットの保存先ディレクトリパスを指定しています。

data_dir = pathlib.Path(DATASET_PATH)     # 指定されたデータセットのパスをpathlibを使用してオブジェクトとして作成します。
if not data_dir.exists():                 # データセットのディレクトリがまだ存在していない場合に処理を実行します。
  tf.keras.utils.get_file(                # TensorFlowのget_fileユーティリティを使用して、指定されたURLからデータセットをダウンロードします。
      'mini_speech_commands.zip',         # ダウンロードされるファイルの名前。
      origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip", # ダウンロード元のURL。
      extract=True,                       # ダウンロード後にZIPファイルを解凍する設定。
      cache_dir='.', cache_subdir='data') # キャッシュの保存先ディレクトリを現在のディレクトリに指定。 キャッシュされたファイルを格納するサブディレクトリを指定
                                          # この場合は、'data'という名前のサブディレクトリが作成され、そこにダウンロードされたファイルが保存されます。
                                          # つまり、./dataディレクトリに、mini_speech_commandsが保存されます

# データセットのオーディオクリップはno 、 yes 、 down 、 go 、 left 、 up 、 right 、 stop に対応する8つのフォルダに保存されます

commands = np.array(tf.io.gfile.listdir(str(data_dir))) # TensorFlowのtf.io.gfile.listdir関数を使用して、指定されたデータディレクトリ（data/mini_speech_commands）内のファイルおよびディレクトリのリストを取得します
commands = commands[commands != 'README.md']            # 取得したディレクトリリストから、'README.md'という要素を取り除きます。
print('Commands:', commands)                            # 最終的なコマンド（またはクラス）のリストを表示します。Commands: ['stop' 'left' 'no' 'go' 'yes' 'down' 'right' 'up']  こんなかんじ

# オーディオクリップをfilenamesというリストに抽出し、シャッフルします。

filenames = tf.io.gfile.glob(str(data_dir) + '/*/*')        # tf.io.gfile.glob関数を使用して、指定されたデータディレクトリ内のすべてのサブディレクトリ内のファイルパスを取得します。パスは ./data/mini_speech_commands/up/music1 のような形式
filenames = tf.random.shuffle(filenames)                    # 取得したファイルパスのリストをランダムにシャッフルします。データセット内の順序がランダムになり、トレーニング時のバッチのバリエーションを確保します。
num_samples = len(filenames)                                # データセット内の総サンプル数を計算します 8000くらい
print('Number of total examples:', num_samples)             # 総サンプル数を表示します。
print('Number of examples per label:',                      # データセット内の各クラス（コマンド）ごとにサンプルの数を表示します。
      len(tf.io.gfile.listdir(str(data_dir/commands[0]))))  # もしcommands[0]が"up"であれば、str(data_dir/commands[0])は'data/mini_speech_commands/up'のような文字列を生成します。
                                                            # commandsが['stop' 'left' 'no' 'go' 'yes' 'down' 'right' 'up']で、それぞれ音のサンプルが1000コずつあれば、'data/mini_speech_commands/up/*のlengthは1000になる
print('Example file tensor:', filenames[0])                 # データセット内の最初のファイルのパスを表示します

# filenamesを、それぞれ80:10:10の比率を使用して、トレーニング、検証、およびテストセットに分割します。

train_files = filenames[:6400]                # ファイルパスのリストから最初の6400個をトレーニングセットとして選択します。[:6400]と[0:6400]は同じ
val_files = filenames[6400: 6400 + 800]       # ファイルパスのリストから6400番目から6400 + 800番目までの800個を検証セットとして選択します。
test_files = filenames[-800:]                 # ファイルパスのリストから末尾から800個をテストセットとして選択します。

print('Training set size', len(train_files))  # トレーニングセットのサイズを表示します。6400
print('Validation set size', len(val_files))  # 検証セットのサイズを表示します。 800
print('Test set size', len(test_files))       # テストセットのサイズを表示します。 800

# オーディオファイルとそのラベルを読む


# テンソル（Tensor）は、数学的には多次元配列を指し、データを表現するための一般的な概念です。ディープラーニングや機械学習の文脈では、テンソルは通常、数値データ（スカラー、ベクトル、行列など）を格納するための多次元配列を指します。
# 次元（Rank）: テンソルの次元は、テンソルが何次元のデータを格納しているかを示します。スカラーは0次元テンソル、ベクトルは1次元テンソル、行列は2次元テンソルといった具体例があります。
# 形状（Shape）: テンソルの形状は、各次元にいくつの要素があるかを示します。例えば、3x4の行列は形状が(3, 4)です。
# データ型（Data Type）: テンソルが格納するデータの型を指定します。例えば、整数、浮動小数点数、文字列などがあります。

test_file = tf.io.read_file(DATASET_PATH+'/down/0a9f9af7_nohash_0.wav')   # tf.io.read_file関数を使用して、指定されたファイルパスにある音声ファイル（ここでは DATASET_PATH+'/down/0a9f9af7_nohash_0.wav'）の内容を読み込みます。
                                                                          # 結果として得られる test_file は、音声ファイルのバイナリデータを含むテンソルです。このテンソルの各要素はバイト（byte）で構成され、音声ファイルが持つ生のバイナリデータを表す。このままでは解釈が難しく、通常はデコードや処理の前段階で使用されます
test_audio, _ = tf.audio.decode_wav(contents=test_file)                   # tf.audio.decode_wav関数を使用して、音声ファイルのバイナリデータをデコードし、テンソルとして取得します。デコードされたテンソルは通常、数値データで構成され、音声ファイルの波形データを表します。
                                                                          # デコードされたテンソルは、音声ファイルの波形を解釈可能な形式に変換されており、機械学習モデルに供給するために一般的に使用されます。test_audioはデコードされた音声データを含むテンソルです。_は、サンプリングレートやビット深度などの補足情報が含まれるが、ここでは無視されています。
test_audio.shape                                                          # test_audioの形状（shape）を表示します。これにより、テンソルの次元（例: サンプル数、チャネル数）が確認できます。TensorShape([13654, 1])のとき、この音声データは13654のサンプルから構成されています。各サンプルは、音声の振幅を表現します。サンプル数は音声データの長さを示します。また、音声データは通常モノラル（1チャネル）であるため1です。
                                                                          # したがって、このテンソルは13654サンプルからなるモノラルの音声データを表しています。各サンプルは波形中の瞬間的な音声の振幅を表し、この形状は機械学習モデルへの入力として適しています。


# モノラル（Monaural）は、音声の録音や再生において、1つのチャネル（1つの音声信号）だけを持つことを指します。対照的に、ステレオ（Stereo）は2つのチャネルを持ち、左右のスピーカーから異なる音声信号が再生されます。

# データセットの生のWAVオーディオファイルをオーディオテンソルに前処理する関数

def decode_audio(audio_binary):
  # Decode WAV-encoded audio files to `float32` tensors, normalized 正規化
  # to the [-1.0, 1.0] range. Return `float32` audio and a sample rate.  元の音声データの振幅を変換して、全体の振幅が範囲[-1.0, 1.0]に収まるようにする操作です。16ビットの音声データの場合、振幅の範囲は[-32768, 32767]です。正規化はこれを一般的に使用される範囲である[-1.0, 1.0]に変換
  audio, _ = tf.audio.decode_wav(contents=audio_binary)               # tf.audio.decode_wav関数を使用して、与えられたバイナリデータ（WAV形式の音声ファイルのバイナリデータ）をデコードします。
  # Since all the data is single channel (mono), drop the `channels`
  # axis from the array.
  return tf.squeeze(audio, axis=-1)                                   # tf.squeeze関数を使用して、audioテンソルからチャネルの次元を削除します。WAV形式の音声ファイルは通常モノラル（1チャネル）であり、デコード後のテンソルもモノラルのため、不要なチャネルの次元を削除します。[13654, 1] -> [13654]
                                                                      # 最終的に、デコードされた音声データが正規化されたfloat32型のテンソルとして返されます。


# 与えられたファイルパスからラベルを取得するための関数

def get_label(file_path):
  parts = tf.strings.split(
      input=file_path,          # windowsならos.path.sep は \ を返す    Unix/Linux/macOSなら/ を返す
      sep=os.path.sep)          # tf.RaggedTensor [[b'data', b'mini_speech_commands', b'up', b'music1']]となる
  # Note: You'll use indexing here instead of tuple unpacking to enable this
  # to work in a TensorFlow graph.
  return parts[-2]              # data/mini_speech_commands/up/music1 だと upが返る


# すべてをまとめる別のヘルパー関数を定義

def get_waveform_and_label(file_path):
  label = get_label(file_path)                # data/mini_speech_commands/up/music1ならup
  audio_binary = tf.io.read_file(file_path)   # file_pathの音声データ読み込み
  waveform = decode_audio(audio_binary)       # テンソルをデコードし、機械学習モデルへの入力に適した形状にする
  return waveform, label                      # 波形データとラベルを返す


# 音声とラベルのペアを抽出するためのトレーニングセットを作成

AUTOTUNE = tf.data.AUTOTUNE # AUTOTUNE は,TensorFlow データパイプラインでの並列処理を効率的に行うための特殊な値

files_ds = tf.data.Dataset.from_tensor_slices(train_files)  # files_dsは、トレーニングデータセット内の各音声ファイルに対応するファイルパスが含まれる TensorFlow データセットです
                                                            # 具体的には、b'data/mini_speech_commands/up/123456.wav',b'data/mini_speech_commands/down/789012.wav'のようなファイルパスが含まれたデータセット

waveform_ds = files_ds.map(
    map_func=get_waveform_and_label,
    num_parallel_calls=AUTOTUNE)            # num_parallel_calls パラメータに AUTOTUNE を指定することで、TensorFlowが最適な並列処理の数を選択し、実行時に調整することができます。
                                            # map メソッドを使用して、各ファイルパスに対して get_waveform_and_label 関数を適用します.
                                            # これにより、各ファイルパスから波形データとラベルのペアを生成するデータセットが作成されます。


# いくつかのオーディオ波形をプロット

rows = 3
cols = 3
n = rows * cols
fig, axes = plt.subplots(rows, cols, figsize=(10, 12))      # 3x3のサブプロットを持つMatplotlibのフィギュア（描画領域）を作成します。つまり枠が9個　各サブプロットは axes リストに格納されます。
                                                            # fig: 作成されたフィギュア全体を表すMatplotlibの Figure オブジェクト axes: サブプロット（軸）の2次元配列です。各サブプロットには、axes[row][col] でアクセスできます。

for i, (audio, label) in enumerate(waveform_ds.take(n)):    # waveform_ds から最初の n サンプルを取得し、各サンプルについて以下の処理を繰り返します。順番はランダムにしてあるので、yes,yes,left,downみたいな感じになる
                                                            # audio,labelは形データとラベルのペア,iがインデックス（何番目か0,1,2...）を表す enumerateを使うと、list内の各要素とそのインデックスを取得できる
  r = i // cols                                             # r = i // cols および c = i % cols で、現在のサンプルのインデックスに基づいて行と列の位置を計算します。
                                                            # 最初ならr = 0 // 3 = 0, c = 0 % 3 = 0より0行0列, 次は r = 1 // 3 = 0, c = 1 % 3 = 1より0行1列みたいな感じ
  c = i % cols                    
  ax = axes[r][c]                                           # ax = axes[r][c]: サブプロットの位置に対応する ax オブジェクト（軸）を取得します。例えば、0行1個目にある描画領域を取得します
  ax.plot(audio.numpy())                                    # サンプルの波形データをプロットします。tensor_audio = tf.constant([0.1, 0.2, 0.3, 0.4], dtype=tf.float32)を[0.1 0.2 0.3 0.4]に変換してる
  ax.set_yticks(np.arange(-1.2, 1.2, 0.2))                  # y軸の目盛り、間隔を設定します。
  label = label.numpy().decode('utf-8')                     # ラベルをバイト文字列からUTF-8でデコードして文字列に変換します。　label = tf.constant(b'My_Label')をデコードすると'My_Label'となる
  ax.set_title(label)                                       # サブプロットのタイトルを設定します。

plt.show()                                                 # Matplotlibで作成したグラフを表示します。


# 波形をスペクトログラムに変換する
# スペクトログラムは、時間の経過に伴って音響信号の周波数成分がどのように変化するかを視覚化するための手法

def get_spectrogram(waveform):
  # Zero-padding for an audio waveform with less than 16,000 samples.
  input_len = 16000
  waveform = waveform[:input_len]                       # 入力の音声波形が16,000サンプル以上の場合、16,000サンプルになるようにする。
  zero_padding = tf.zeros(
      [16000] - tf.shape(waveform),
      dtype=tf.float32)                                 #  不足している部分をゼロで埋めるためのゼロパディングを行います。音声波形の長さが16,000未満の場合、足りない部分をゼロで埋めています。
                                                        # これをしないと入力の音声波形が16,000サンプル以下のまま素通りするので、モデルサイズが可変となり扱いにくい
  # Cast the waveform tensors' dtype to float32.
  waveform = tf.cast(waveform, dtype=tf.float32)        # 音声波形のデータ型は元々 tf.float32 だが、明示的に書く。
  # Concatenate the waveform with `zero_padding`, which ensures all audio
  # clips are of the same length.
  equal_length = tf.concat([waveform, zero_padding], 0) # ゼロパディングされた部分を音声波形に結合して、すべての音声クリップが同じ長さになるようにします
                                                        # zero_padding = tf.zeros([16000] - tf.shape(waveform), dtype=tf.float32)は、単にゼロで埋められたテンソルを生成するだけであり、この段階では元の音声波形には結合されていません。
                                                        # equal_length = tf.concat([waveform, zero_padding], 0)は、waveform と zero_padding を垂直方向に結合して、同じ長さにする操作です。
                                                        # この操作により、ゼロで埋められた部分が waveform に追加され、結果として equal_length が得られます。
                                                        # つまり、zero_padding はゼロで埋めるだけであり、実際に音声波形に結合するためには tf.concat が必要です。この操作がないと、ゼロパディングはただのゼロのテンソルであり、元の音声波形には影響を与えません
  # Convert the waveform to a spectrogram via a STFT.
  spectrogram = tf.signal.stft(                         # STFT は信号を時間と周波数の2次元グラフに変換する手法
      equal_length, frame_length=255, frame_step=128)   # 等長になった波形に対してショートタイムフーリエ変換（STFT）を適用して、音声データからスペクトログラムを得ます。
                                                        # frame_length は各フレームのサンプル数であり、frame_step はフレーム間のサンプル数のステップサイズです
                                                        # STFT は信号を小さなフレームに区切り、各フレームに対してフーリエ変換を行います
                                                        # frame_length=256 とすると、元の信号が256サンプルのフレームに分割されます。
                                                        # フレームはオーバーラップすることがあり、次のフレームは前のフレームの一部を含むことになります。
                                                        # frame_step=128 とすると、次のフレームは前のフレームから128サンプルずれて開始されます。
                                                        # ステップサイズが小さいほどフレーム同士がオーバーラップし、時間方向の情報が重複します

  #例えば、Signal: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]のシーケンスを、frame_length=4 および frame_step=2 でSTFTにかけると、
  # フレーム1: [0, 1, 2, 3], フレーム2: [2, 3, 4, 5], フレーム3: [4, 5, 6, 7]となる
                                                      
  # Obtain the magnitude of the STFT.
  spectrogram = tf.abs(spectrogram)                     # STFTの結果から振幅情報を取得します。
  # Add a `channels` dimension, so that the spectrogram can be used
  # as image-like input data with convolution layers (which expect
  # shape (`batch_size`, `height`, `width`, `channels`).
  spectrogram = spectrogram[..., tf.newaxis]            # スペクトログラムに新たな次元を追加して、（バッチサイズ、高さ、幅、チャネル）の形状を持つテンソルに変換します
                                                        # 具体的には、spectrogram は既に計算されたスペクトログラムで、その形状は (time, frequency) です。この操作によって、新しい次元が最後に追加され、形状は (time, frequency, 1) となります。
                                                        # CNNにスペクトログラムを入力する飲み適した形状にするため。通常は (time, frequency, channels) の形状が期待される。ただし、音声データは通常モノラルであるため、channels は 1。
                                                        # tf.newaxis は、新しい次元を表現するための特別な定数です。これは単なる None と同等で、新しい次元の位置に置き換えられます。
                                                        # tensor = tf.constant([[1, 2], [3, 4]])にnew_dimension = tensor[..., tf.newaxis]したら、new_dimension は[[[1], [2]], [[3], [4]]]になる
  return spectrogram


# 1つの例のテンソル化された波形と対応するスペクトログラムの形状を印刷し、元のオーディオを再生

for waveform, label in waveform_ds.take(1):             # データセットから1つの要素（波形とラベル）を取得しています。
  label = label.numpy().decode('utf-8')                 # ラベルをNumPy配列に変換して文字列にデコードしています。yesとか
  spectrogram = get_spectrogram(waveform)               # 取得した波形データを使用して、get_spectrogram 関数を呼び出してスペクトログラムを計算しています。

print('Label:', label)                                  # データのクラスまたはカテゴリのラベル（文字列）Label: yes
print('Waveform shape:', waveform.shape)                # 元の波形の形状 Waveform shape: (16000,)
print('Spectrogram shape:', spectrogram.shape)          # 計算されたスペクトログラムの形状 Spectrogram shape: (124, 129, 1)
print('Audio playback')                                 # 波形データの再生を可能にするオーディオプレーヤーを表示しています。
display.display(display.Audio(waveform, rate=16000))    # display.Audio(waveform, rate=16000): この部分は、waveform という音声データを持ち、サンプリングレートが16000 Hzであることを示しています。
                                                        # サンプリングレート（Sampling Rate）は、アナログ信号をデジタル化する際に、アナログ信号から一定の間隔でサンプルを取得する頻度を表します。
                                                        # サンプリングレートが16,000 Hzであれば、1秒間に16,000回のサンプリングが行われていることを意味します


# スペクトログラムを表示するための関数を定義

def plot_spectrogram(spectrogram, ax):
  if len(spectrogram.shape) > 2:
    assert len(spectrogram.shape) == 3                              # スペクトログラムの次元が3次元以上の場合かつ、3次元の場合、最後の次元が1であることが期待されます。不要な次元を取り除くために np.squeeze を使用します。
    spectrogram = np.squeeze(spectrogram, axis=-1)                  # np.array([[[1]], [[2]], [[3]]]) -> arr_squeezed = np.squeeze(arr) =>  [1 2 3]　サイズ1の次元を削除
                                                                    # spectrogram = spectrogram[..., tf.newaxis]で次元を増やしたのは CNN への入力形状への対応のためであり、その後に不要な次元をnp.squeeze(spectrogram, axis=-1)で削除して元の形状に戻しています。
  # Convert the frequencies to log scale and transpose, so that the time is
  # represented on the x-axis (columns).
  # Add an epsilon to avoid taking a log of zero.
  log_spec = np.log(spectrogram.T + np.finfo(float).eps)            # スペクトログラムの各要素に対して自然対数を取ります。対数変換は、人間の聴覚に合わせて音の強さを表現するのに一般的です。np.finfo(float).eps は、計算安定性のために微小な値を追加しています。
                                                                    # spectrogram.T: スペクトログラムを転置しています。これにより、周波数が横軸に、時間が縦軸に配置されます。
                                                                    # np.finfo(float).eps: 浮動小数点数型 (float) における最小の正の値を返す NumPy の関数です。この値は非常に小さな正の値で、計算の安定性を保つために使用されます。
                                                                    # spectrogram.T + np.finfo(float).eps: スペクトログラムの転置行列の各要素に np.finfo(float).eps を加算しています。
                                                                    # 対数変換の際にゼロや非常に小さな値が現れた場合に、対数が発散してしまうのを防ぐために行われます。 
  height = log_spec.shape[0]                                        # 時間が縦軸になり、各時間点での周波数成分が横軸になります。
  width = log_spec.shape[1]                                         # スペクトログラムの高さと幅を取得します。
  X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)    # スペクトログラムの横軸（時間）に対するデータを作成します。print(np.linspace(0, 10, 3)) -> [ 0.  5. 10.] linspace 等差数列を生成する
  Y = range(height)                                                 # スペクトログラムの縦軸（周波数）に対するデータを作成します。
  ax.pcolormesh(X, Y, log_spec)                                     # pcolormesh を使用して、スペクトログラムをカラーメッシュとしてプロットします。これにより、スペクトログラムの強度が色で表現されます。


# 時間の経過に伴う例の波形と対応するスペクトログラムをプロット

fig, axes = plt.subplots(2, figsize=(12, 8))     # 2つのサブプロットを持つ図を作成
timescale = np.arange(waveform.shape[0])         # print(np.arange(3)) -> [0 1 2], print(np.arange(3, 10)) -> [3 4 5 6 7 8 9]
axes[0].plot(timescale, waveform.numpy())        # 1つ目のサブプロットに波形をプロットします。横軸に時間(timescale)、縦軸に波形の振幅(waveform)が配置されます。
axes[0].set_title('Waveform')                    # タイトル 'Waveform' を設定
axes[0].set_xlim([0, 16000])                     # 横軸の表示範囲を0から16000までに設定

plot_spectrogram(spectrogram.numpy(), axes[1])   #  2つ目のサブプロットにスペクトログラムをプロットするためのヘルパー関数 plot_spectrogram を呼び出します
axes[1].set_title('Spectrogram')                 # 2つ目のサブプロットにタイトル 'Spectrogram' を設定
plt.show()                                       # 図を表示


# 波形データセットをスペクトログラムとそれに対応するラベルに整数IDとして変換する関数を定義

def get_spectrogram_and_label_id(audio, label):
  spectrogram = get_spectrogram(audio)
  label_id = tf.argmax(label == commands)
  return spectrogram, label_id

