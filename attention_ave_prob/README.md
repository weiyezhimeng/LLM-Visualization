## How to Run
```sh
$ cd wow
$ conda activate zhaojiawei_attention
$ 修改main.py中 ```device = "cuda:7"```
$ python main.py
在服务器上部署代码后运行，一般来说会转发到本地的5000端口
```
 - 打开本地浏览器，访问 `http://127.0.0.1:5000/static/index.html`
 - 输入模型的绝对路径，prompt，点击submit
 - 等待片刻即可
