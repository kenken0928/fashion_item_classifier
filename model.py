
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

# 正解ラベルのリストを設定
# 日本語
classes_ja = ['Tシャツ/トップ', 'ズボン', 'プルオーバー', 'ドレス', 'コート', 'サンダル', 'ワイシャツ', 'スニーカー', 'バッグ', 'アンクルブーツ']

# 英語
classes_en = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Angle_boot']

# 正解ラベルの数
number_class = len(classes_ja)

# 画像のピクセル
image_size = 28


# 画像認識のモデル
# クラスの定義
class Net(nn.Module):

  # ーー 初期設定 ーー
  def __init__(self):

    # nn.Moduleクラスのオーバーライド
    super().__init__()

    # 畳み込み層-1 
    self.conv_1 = nn.Conv2d(1, 8, 3)
    self.conv_2 = nn.Conv2d(8, 16, 3)
    self.batchnorm_1 = nn.BatchNorm2d(16)

    # 畳み込み層-2
    self.conv_3 = nn.Conv2d(16, 32, 3)
    self.conv_4 = nn.Conv2d(32, 64, 3)
    self.batchnorm_2 = nn.BatchNorm2d(64)

    # プーリング層
    self.pool = nn.MaxPool2d(2, 2)

    # 活性化関数
    self.relu = nn.ReLU()

    # 全結合層-1
    self.fc_1 = nn.Linear(64*4*4, 256)

    # ドロップアウト層
    self.dropout = nn.Dropout(p=0.5)

    # 全結合層-2（出力層）
    self.fc_2 = nn.Linear(256, 10)


  # ーー 順伝播 ーー
  def forward(self, x):

    # 畳み込み層-1
    x = self.relu(self.conv_1(x))
    x = self.relu(self.batchnorm_1(self.conv_2(x)))
    x = self.pool(x)

    # 畳み込み層-2
    x = self.relu(self.conv_3(x))
    x = self.relu(self.batchnorm_2(self.conv_4(x)))
    x = self.pool(x)

    # Flatten層
    x = x.view(-1, 64*4*4)

    # 全結合層-1
    x = self.relu(self.fc_1(x))

    # ドロップアウト層
    x = self.dropout(x)

    # 全結合層-2
    x = self.fc_2(x)

    return x


# クラスのインスタンス生成
net = Net()

# GPU対応
# Webアプリでの実行では、GPUは不要
# net.cuda()


# 訓練済みパラメータの読み込みと設定
"""
net.load_state_dict():
読み込んだパラメータを、net という名前のニューラルネットワークモデルのパラメータに適用する。
モデルのアーキテクチャや構造は別途定義されている必要あり。

torch.load('ファイルパス'):
ファイルの読み込み

map_location=torch.device('cpu'):
GPU上にモデルが保存されている場合でも、CPU上に読み込むための指定
"""
net.load_state_dict(torch.load('model_cnn.pth', map_location=torch.device('cpu')))


# ーーーー 予測 ーーーー
def predict(img):
  
  # ーー モデルへの入力 ーー
  # モノクロに変換
  img = img.convert('L')

  # 画像のピクセルデータのサイズを変更
  # データの形状が変わる訳ではないのに注意。データの形状を変えるのは下記の、.reshape(1, 1, image_size, image_size) にて。
  img = img.resize((image_size, image_size))

  # 入力された画像データの前処理の定義
  transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.0), (1.0))])

  # 前処理の実行
  img = transform(img)

  # リサイズ
  """
  一般的に、CNNモデルに投入する画像データは、（バッチサイズ、チャネル数、画像の高さ、画像の幅）
  """
  x = img.reshape(1, 1, image_size, image_size)

  # predict
  # ーー 評価モード ーー
  net.eval()

  # 順伝播
  y = net(x)

  # 結果を確率で取得
  # torch.squeeze(y) → yの各要素が確率と解釈され、合計が1になるように正規化される。
  y_prob = torch.nn.functional.softmax(torch.squeeze(y))


  # 降順にソート（値が大きいもの → 小さいもの、の順）
  sorted_prob, sorted_indices = torch.sort(y_prob, descending=True)


  # リスト形式でreturn
  return [(classes_ja[idx], classes_en[idx], prob.item()) for idx, prob in zip(sorted_indices, sorted_prob)]
