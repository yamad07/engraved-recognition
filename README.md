# engraved-recognition

## Usage
1.anaconda環境を構築
[参考](https://qiita.com/sk427/items/9f215931c8249ada75cd)

2.下のライブラリをインストール。

3.サーバーを起動

```
$ cd server
$ FLASK_APP=main.py flask run
```

4.別ウィンドウでサンプルの写真を使ってcurlコマンドで叩く
```
curl -X POST http://localhost:5000/recognition -F "img=@./sample.jpg"
```

## Library Version
anaconda3-4.4.0
flask 0.12.2
flask_restful
pytorch 0.3.0.post4
torchvision 0.2.0
opencv 3.1.0
