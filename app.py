
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image

from model import predict     # 上記で用意した model.py の 関数 predict


# ファイルアップロード時の警告メッセージを非表示にする設定
st.set_option('deprecation.showfileUploaderEncoding', False)


# サイドバーのタイトル表示設定
st.sidebar.title('画像認識アプリ')

# サイドバーへのテキスト表示設定
st.sidebar.write('オリジナルの画像認識モデルを使って、何の画像かを、判定します。')

# サイドバーへ、空の行を追加
st.sidebar.write('')


# 画像のアップロード方法に関する「ラジオボタン」の設定
image_source = st.sidebar.radio('画像のソースを選択してください。', ('画像をアップロード', 'カメラで撮影'))

# ローカルから画像をアップロードする場合
if image_source == '画像をアップロード':

  # 画像選択UIの設定
  image_file = st.sidebar.file_uploader('画像を選択してください。', type=['png', 'jpg', 'jpeg'])


# カメラで撮影する場合
elif image_source == 'カメラで撮影':

  # カメラによる撮影UIの設定
  image_file = st.camera_input('カメラで撮影')



# ユーザーが画像をアップロード or カメラで撮影した場合のみ、以下を実行
if image_file is not None:

  # 「ローディングアイコン」を表示させる設定
  with st.spinner('推定中・・・'):

    # PILライブラリーを使用し、選択された画像ファイルを開き、img に代入
    img = Image.open(image_file)

    # st.image()を使用し、img に格納された画像を表示
    # width で、表示幅を指定
    st.image(img, caption='対象の画像', width=480)

    # 空の行を追加
    st.write('')


    # ーーーー 予測(predict) ーーーー
    # 上記で設定した model.pyのpredict(img)関数の実行
    # results に、降順にソートされた予測値の確率のリストを代入
    # results : (日本語のラベル、英語のラベル、確率）
    results = predict(img)


    # ーーーー 結果の表示 ーーーー
    st.subheader('判定結果')

    # 確率が高い順に、3位まで返す
    number_top = 3

    # Top3に対する繰り返し処理
    for result in results[:number_top]:

      # 結果をテキスト表示
      st.write(str(round(result[2]*100, 2)) + '%の確率で' + result[0] + 'です')
    

    # 円グラフの表示（英語表記）
    pie_labels = [result[1] for result in results[:number_top]]

    # 円グラフにTop3以下用に、othersを追加
    pie_labels.append('the others')


    # Top3のラベルを円グラフに表示
    pie_probs = [result[2] for result in results[:number_top]]

    # Top3以降のラベルをothersにまとめて、円グラフに表示
    pie_probs.append(sum([result[2] for result in results[number_top:]]))

    # 描画設定
    fig, ax = plt.subplots()

    # 円グラフの描画設定
    wedgeprops = {'width':0.3, 'edgecolor':'white'}
    textprops = {'fontsize':6}

    # 描画
    ax.pie(pie_probs, labels=pie_labels, counterclock=False, startangle=90, textprops=textprops, autopct='%.2f', wedgeprops=wedgeprops)

    # plt.show()みたいなもの
    st.pyplot(fig)


st.sidebar.write('')
st.sidebar.write('')

st.sidebar.caption("""
このアプリは、「Fashion-MNIST」を訓練データとして使っています。\n
""")
