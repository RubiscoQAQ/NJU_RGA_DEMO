from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

import vo.answerJson

system_prompt = (
    '''
    现在要你帮忙将一段用户输入的文本转换为Json格式用于保存。
    规则：
    如果输入为空，则直接返回：{{"content":"我不知道","cite": []"]}}
    (1). 用户输入的文本会包含 [回答] 和 [参考文献] 两个标签。分别对应json中的content和cite两个key。你需要将[回答]标签后面的内容作为content的value，[参考文献]标签后面的内容作为cite的value
    (2). content对应的值是一个字符串，存储了用户输入的回答内容部分
    (3). cite对应的值是一个字符串列表，存储了用户输入参考文献内容部分。你需要正确的将用户输入的参考文献文本拆分成列表
    (4). 输出格式规定为Json格式，例如：{{"content":"content_string","cite": ["cite1_string", "cite2_string"]}}
    现在有两个角色：
    - Json转换器，精通自然语言处理。用于将用户输入按照规则转换为Json格式。
    - Json校验器，擅长检测Json语法。用于检查Json格式是否合法。
    你需要按照以下步骤完成工作，且每一步都要遵守以上规则：
    Step 1: 扮演Json转换器，按照规则，将用户输入转换为目标Json格式，例如：{{"content":"content_string","cite": ["cite1_string", "cite2_string"]}}，然后执行Step 2
    Step 2: 扮演Json校验器，按照Json语法，校验上一步的输出是否满足目标格式：{{"content":"content_string","cite": ["cite1_string", "cite2_string"]}}。如果不满足，则返回Step 1，如果满足，则进入Step 3
    Step 3: 输出经过Step 2校验过的Json格式文件
    说明：content_string、cite1_string、cite2_string、回答内容、参考文献内容 为占位符，你不能直接输出
    约束：你最终只能输出一个Json格式的字符串。不要有其他任何无关输出。
    重要约束：输出时，如果有双引号，将它替换为单引号
    '''
)
user_input = (
    '''
    \n\n
    用户输入文本如下：
    {unformat_out}
    '''
)
parser = JsonOutputParser(
    pydantic_object=vo.answerJson.OutJson
)
formatPrompt = PromptTemplate(
    template=system_prompt+user_input,
    input_variables=['unformat_out'],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

formatPrompt.pretty_print()

