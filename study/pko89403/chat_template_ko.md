# 채팅 템플릿

채팅 템플릿은 언어 모델과 사용자 간 상호작용을 구조화하는 데 필수적입니다. 이는 대화의 형식을 일관되게 유지하고, 모델이 각 메시지의 맥락과 역할을 이해하여 적절한 응답 패턴을 유지할 수 있도록 돕습니다.

## 기본 모델 vs 명령형 모델(Instruct Model)

기본 모델(Base Model)은 다음 토큰을 예측하기 위해 원시 텍스트 데이터로 학습된 반면, 명령형 모델(Instruct Model)은 명령을 따르고 대화를 수행하도록 세부적으로 튜닝됩니다. 예를 들어, `SmolLM2-135M`은 기본 모델이고, `SmolLM2-135M-Instruct`는 이를 명령형으로 튜닝한 변형입니다.

기본 모델을 명령형 모델처럼 동작하게 만들려면, 모델이 이해할 수 있는 일관된 방식으로 프롬프트를 형식화해야 합니다. 여기서 채팅 템플릿이 중요한 역할을 합니다. ChatML은 대화에 역할 표시(시스템, 사용자, 어시스턴트)를 명확히 제공하는 대표적인 템플릿 형식입니다.

기본 모델은 다양한 채팅 템플릿으로 세부 튜닝될 수 있으므로, 명령형 모델을 사용할 때는 적절한 채팅 템플릿을 사용하는 것이 중요합니다.

## 채팅 템플릿 이해하기

본질적으로, 채팅 템플릿은 언어 모델과 소통할 때 대화가 어떻게 형식화되어야 하는지 정의합니다. 여기에는 시스템 수준의 지시, 사용자 메시지, 어시스턴트 응답이 포함되며, 구조적인 형식으로 모델이 이해할 수 있도록 돕습니다. 이 구조는 상호작용의 일관성을 유지하고, 다양한 입력 유형에 대해 모델이 적절히 응답하도록 보장합니다. 아래는 채팅 템플릿 예제입니다:

```sh
<|im_start|>user
Hi there!<|im_end|>
<|im_start|>assistant
Nice to meet you!<|im_end|>
<|im_start|>user
Can I ask a question?<|im_end|>
<|im_start|>assistant
```

`transformers` 라이브러리는 모델의 토크나이저와 관련하여 채팅 템플릿을 자동으로 처리합니다. `transformers`에서 채팅 템플릿이 어떻게 작동하는지 자세히 알아보려면 [여기](https://huggingface.co/docs/transformers/en/chat_templating#how-do-i-use-chat-templates)를 참조하세요. 우리가 해야 할 일은 메시지를 올바르게 구조화하는 것이며, 토크나이저가 나머지를 처리합니다. 다음은 기본 대화 예제입니다:

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant focused on technical topics."},
    {"role": "user", "content": "Can you explain what a chat template is?"},
    {"role": "assistant", "content": "A chat template structures conversations between users and AI models..."}
]
```

위 예제를 살펴보며, 채팅 템플릿 형식과의 연관성을 확인해 봅시다.

## 시스템 메시지

시스템 메시지는 모델의 행동 방식을 설정하는 기본 역할을 합니다. 이는 지속적인 지침으로 작용하며 이후 모든 상호작용에 영향을 미칩니다. 예를 들어:

```python
system_message = {
    "role": "system",
    "content": "You are a professional customer service agent. Always be polite, clear, and helpful."
}
```

## 대화

채팅 템플릿은 대화 기록을 통해 컨텍스트를 유지하며, 사용자와 어시스턴트 간의 이전 교환을 저장합니다. 이를 통해 다중 턴 대화에서도 더 일관된 응답을 제공합니다:

```python
conversation = [
    {"role": "user", "content": "I need help with my order"},
    {"role": "assistant", "content": "I'd be happy to help. Could you provide your order number?"},
    {"role": "user", "content": "It's ORDER-123"},
]
```

## Transformers 라이브러리로 구현

`transformers` 라이브러리는 채팅 템플릿을 기본적으로 지원합니다. 다음은 사용하는 방법입니다:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")

messages = [
    {"role": "system", "content": "You are a helpful coding assistant."},
    {"role": "user", "content": "Write a Python function to sort a list"},
]

# 채팅 템플릿 적용
formatted_chat = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
```

## 사용자 정의 형식

다양한 메시지 유형에 대해 사용자 정의 형식을 추가할 수 있습니다. 예를 들어, 역할별로 특별한 토큰이나 형식을 추가할 수 있습니다:

```python
template = """
<|system|>{system_message}
<|user|>{user_message}
<|assistant|>{assistant_message}
""".lstrip()
```

## 다중 턴 지원

템플릿은 다중 턴 대화를 처리하며 컨텍스트를 유지할 수 있습니다:

```python
messages = [
    {"role": "system", "content": "You are a math tutor."},
    {"role": "user", "content": "What is calculus?"},
    {"role": "assistant", "content": "Calculus is a branch of mathematics..."},
    {"role": "user", "content": "Can you give me an example?"},
]
```

⏭️ [다음: Supervised Fine-Tuning](./supervised_fine_tuning.md)

## 참고 자료

- [Hugging Face Chat Templating Guide](https://huggingface.co/docs/transformers/main/en/chat_templating)
- [Transformers Documentation](https://huggingface.co/docs/transformers)
- [Chat Templates Examples Repository](https://github.com/chujiezheng/chat_templates) 