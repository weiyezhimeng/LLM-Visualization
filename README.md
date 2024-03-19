# attention

为了更好地研究LLM推理的因果关系，我在`https://github.com/mattneary/attention`的基础上
 - 适配了更多的大模型
 - 加入了模型不同layer，不同head的attention
 - 仅支持选择单个token

# attention_ave

由于一次性上传了所有结果，长文本会导致attention内存爆炸，下次修订时会修改交互逻辑，使每一次选择文本时仅上传相应部分attention。因此上传attention_ave，基本和`https://github.com/mattneary/attention`一致。

## How to Run
```sh
$ cd attention
$ pip install poetry
//说实话在https://github.com/mattneary/attention之前都不知道poetry是什么。。。
$ poetry install
cd attention/wow
$ python main.py
在服务器上部署代码后运行，一般来说会转发到本地的5000端口
```
 - 首先在main中输入prompt（鉴于转义字符，没有在前端写输入框）
 - 打开本地浏览器，访问 `http://127.0.0.1:5000/static/index.html`
 - 输入模型的绝对路径，点击submit
 - 等待片刻即可
