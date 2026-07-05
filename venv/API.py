import requests
import json
import base64

# ========== 你的API配置 ==========
APP_KEY = "sk-xuanji-2026525296-T1VZV2pHS3hJVXpZcXZOZg=="
MODEL = "Volc-DeepSeek-V3.2"  # 可选: Volc-DeepSeek-V3.2, Doubao-Seed-2.0-pro, qwen3.5-plus


# =================================

def call_vivo_llm(user_message, system_prompt=None, stream=False):
    """
    调用vivo大模型API
    :param user_message: 用户输入的内容（可以是文本，也可以是语音转文字后的结果）
    :param system_prompt: 系统提示词（设定角色、任务等）
    :param stream: 是否流式输出
    :return: 模型返回的完整回答
    """
    url = "https://api-ai.vivo.com.cn/v1/chat/completions"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {APP_KEY}"
    }

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_message})

    data = {
        "model": MODEL,
        "messages": messages,
        "stream": stream,
        "max_tokens": 4096,
        "temperature": 0.7,
        "reasoning_effort": "medium"  # 适中的思考深度
    }

    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        result = response.json()
        return result.get("choices", [{}])[0].get("message", {}).get("content", "")
    else:
        print(f"API调用失败: {response.status_code}")
        print(response.text)
        return None


# ========== 示例用法 ==========
if __name__ == "__main__":
    # 模拟：用户说了一段话（语音转文字后的结果）
    user_text = "我今天早上吃了稀饭和鸡蛋，然后出门散步，但是我想不起来我把钥匙放哪了。"

    system_prompt = """你是一位专业的老年认知评估助手。请根据用户的描述，分析以下几点：
    1. 语言流畅度（是否流畅、有无卡顿）
    2. 逻辑连贯性（叙述是否有条理）
    3. 记忆表现（是否出现明显遗忘描述）
    4. 情绪状态（语气积极还是焦虑）
    请以结构化JSON格式输出分析结果。
    """

    result = call_vivo_llm(user_text, system_prompt)
    print("大模型分析结果：")
    print(result)