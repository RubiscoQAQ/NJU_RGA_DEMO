# 项目结构
- config包用于配置
- documentLoader包，用于解决在windows环境下无法通过pip引入jq，导致官方JsonLoader不可用的问题
- prompts包，用于存放定义的prompt
- utils包，工具类，实际没太用上
- vo包，定义了json格式，实际似乎用处不大
- demo.py 开发时用于debug
- main.py
  - user_ask:用于回答用户提供的问题
  - jsonl：用于解答jsonl的问题，并输出
- rga.py 封装的RGA类
# 项目开发
见`RGA开发文档.md`