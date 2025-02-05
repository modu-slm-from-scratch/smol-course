
# Chat Templates

Chat template은 언어 모델과 사용자 간의 상호작용을 구조화하는 데 필수적입니다. 모델이 각 메시지의 맥락과 역할을 이해하면서 적절한 응답 패턴을 유지할 수 있도록 일관된 대화 형식을 제공합니다.

## Base Models vs Instruct Models

Base model은 다음 토큰을 예측하기 위해 원시 텍스트 데이터로 학습된 모델이며, Instruct model은 지시사항을 따르고 대화에 참여하도록 특별히 미세 조정된 모델입니다. 예를 들어, `SmolLM2-135M`은 base model이고, `SmolLM2-135M-Instruct`는 이를 instruction-tuning한 변형입니다.

Base model을 instruct model처럼 작동하게 만들기 위해서는 모델이 이해할 수 있는 일관된 방식으로 프롬프트를 포맷해야 합니다. 이것이 chat template이 필요한 이유입니다. ChatML은 명확한 역할 지시자(system, user, assistant)로 대화를 구조화하는 템플릿 형식 중 하나입니다.

Base model이 서로 다른 chat template로 미세 조정될 수 있다는 점을 주목해야 합니다. 따라서 instruct model을 사용할 때는 올바른 chat template을 사용하고 있는지 확인해야 합니다.

## Chat Templates 이해하기

Chat template의 핵심은 언어 모델과 통신할 때 대화를 어떻게 포맷해야 하는지 정의하는 것입니다. 여기에는 시스템 수준의 지시사항, 사용자 메시지, 그리고 모델이 이해할 수 있는 구조화된 형식의 assistant 응답이 포함됩니다. 이 구조는 상호작용 전반에 걸쳐 일관성을 유지하고 모델이 다양한 유형의 입력에 적절하게 응답하도록 보장합니다. 다음은 chat template의 예시입니다:

```sh
<|im_start|>user
Hi there!<|im_end|>
<|im_start|>assistant
Nice to meet you!<|im_end|>
<|im_start|>user
Can I ask a question?<|im_end|>
<|im_start|>assistant
```

`transformers` 라이브러리는 모델의 tokenizer와 관련하여 chat template을 자동으로 처리합니다. Transformers가 어떻게 chat template을 구성하는지에 대해 [여기](https://huggingface.co/docs/transformers/en/chat_templating#how-do-i-use-chat-templates)에서 자세히 알아볼 수 있습니다. 우리가 해야 할 일은 메시지를 올바른 방식으로 구성하는 것뿐이며, tokenizer가 나머지를 처리할 것입니다. 다음은 기본적인 대화의 예시입니다:

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant focused on technical topics."},
    {"role": "user", "content": "Can you explain what a chat template is?"},
    {"role": "assistant", "content": "A chat template structures conversations between users and AI models..."}
]
```

위의 예시를 분석하고 chat template 형식에 어떻게 매핑되는지 살펴보겠습니다.

## System Messages

System message는 모델이 어떻게 행동해야 하는지에 대한 기초를 설정합니다. 이는 이후의 모든 상호작용에 영향을 미치는 지속적인 지시사항으로 작용합니다. 예시:

```python
system_message = {
    "role": "system",
    "content": "You are a professional customer service agent. Always be polite, clear, and helpful."
}
```

## Conversations

Chat template은 대화 기록을 통해 맥락을 유지하며, 사용자와 assistant 간의 이전 대화를 저장합니다. 이를 통해 더 일관성 있는 다중 턴 대화가 가능합니다:

```python
conversation = [
    {"role": "user", "content": "I need help with my order"},
    {"role": "assistant", "content": "I'd be happy to help. Could you provide your order number?"},
    {"role": "user", "content": "It's ORDER-123"},
]
```

## Transformers를 이용한 구현

Transformers 라이브러리는 chat template에 대한 내장 지원을 제공합니다. 사용 방법은 다음과 같습니다:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")

messages = [
    {"role": "system", "content": "You are a helpful coding assistant."},
    {"role": "user", "content": "Write a Python function to sort a list"},
]

# Chat template 적용
formatted_chat = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
```

## 커스텀 포맷팅

다른 메시지 유형의 포맷 방식을 커스터마이즈할 수 있습니다. 예를 들어, 다른 역할에 대한 특수 토큰이나 포맷을 추가할 수 있습니다:

```python
template = """
<|system|>{system_message}
<|user|>{user_message}
<|assistant|>{assistant_message}
""".lstrip()
```

## Multi-Turn 지원

Template은 맥락을 유지하면서 복잡한 다중 턴 대화를 처리할 수 있습니다:

```python
messages = [
    {"role": "system", "content": "You are a math tutor."},
    {"role": "user", "content": "What is calculus?"},
    {"role": "assistant", "content": "Calculus is a branch of mathematics..."},
    {"role": "user", "content": "Can you give me an example?"},
]
```

⏭️ [다음: Supervised Fine-Tuning](./supervised_fine_tuning.md)

## 리소스

- [Hugging Face Chat Templating 가이드](https://huggingface.co/docs/transformers/main/en/chat_templating)
- [Transformers 문서](https://huggingface.co/docs/transformers)
- [Chat Templates Examples Repository](https://github.com/chujiezheng/chat_templates)
