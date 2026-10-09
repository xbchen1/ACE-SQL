"""Response parsing shared by benchmark inference stages."""
import re


def parse_retriever_response(response: str) -> str:
    """
    解析检索器响应，提取列列表或JSON格式

    Args:
        response: 模型生成的响应文本

    Returns:
        解析后的列列表字符串或JSON字符串
    """
    # 方法1：提取 ```json ... ``` 代码块中的JSON对象
    json_block_pattern = re.compile(r'```json\s*\n?(.*?)\n?```', re.DOTALL)
    match = json_block_pattern.search(response)
    if match:
        return match.group(1).strip()

    # 方法2：提取代码块中的列表
    code_block_pattern = re.compile(r'```(?:python|sql|text)?\s*\n?\s*(\[[^\]]*\])\s*\n?```', re.DOTALL)
    match = code_block_pattern.search(response)
    if match:
        return match.group(1)

    # 方法3：普通列表格式
    list_pattern = re.compile(r'\[([^\]]+)\]')
    match = list_pattern.search(response)
    if match:
        return f"[{match.group(1)}]"

    # 方法4：返回原始响应（让后续处理来解析）
    return response


def parse_sql_response(response: str) -> str:
    """
    解析SQL响应，提取SQL语句

    Args:
        response: 模型生成的响应文本

    Returns:
        解析后的SQL语句
    """
    if not response:
        return ""

    # 方法1：提取```sql代码块
    matches = re.findall(r'```sql\s*(.*?)\s*```', response, re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[-1].strip()

    # 方法2：提取普通代码块
    matches = re.findall(r'```\s*(.*?)\s*```', response, re.DOTALL)
    if matches:
        sql_candidate = matches[-1].strip()
        if any(keyword in sql_candidate.upper() for keyword in ['SELECT', 'INSERT', 'UPDATE', 'DELETE']):
            return sql_candidate

    # 方法3：查找SELECT语句
    if 'SELECT' in response.upper():
        lines = response.split('\n')
        sql_lines = []
        in_sql = False
        for line in lines:
            if 'SELECT' in line.upper():
                in_sql = True
            if in_sql:
                sql_lines.append(line)
                if ';' in line:
                    break
        if sql_lines:
            return '\n'.join(sql_lines).strip()

    # 方法4：返回原始响应
    return response.strip()
