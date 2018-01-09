# engraved-recognition

## Usage
1.anaconda環境を構築する
[参考](https://qiita.com/sk427/items/9f215931c8249ada75cd)

2.仮想環境を作る

```
$ conda env create --file env.yml
```

3.仮想環境に入る

```
$ source activate recarat-flask-api
```
4.サーバーを起動する

```
$ cd server
$ FLASK_APP=main.py flask run
```

5.別ウィンドウでサンプルの写真を使ってcurlコマンドで叩く
```
curl -X POST http://localhost:5000/recognition -F "img=@./sample.jpg"
```
