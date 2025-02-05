# Supervised Fine-Tuning

Supervised Fine-Tuning (SFT)는 사전 학습된 언어 모델을 특정 작업이나 도메인에 맞게 조정하는 중요한 프로세스입니다. 사전 학습된 모델들이 인상적인 일반적 능력을 가지고 있지만, 특정 사용 사례에서 뛰어난 성능을 내기 위해서는 커스터마이징이 필요한 경우가 많습니다. SFT는 사람이 검증한 예시들로 구성된 신중하게 선별된 데이터셋으로 모델을 추가 학습함으로써 이러한 간극을 메웁니다.

## Supervised Fine-Tuning 이해하기

Supervised fine-tuning의 핵심은 레이블이 지정된 토큰의 예시들을 통해 사전 학습된 모델에게 특정 작업을 수행하도록 가르치는 것입니다. 이 과정은 모델에게 원하는 입력-출력 동작의 많은 예시를 보여줌으로써 사용 사례에 특화된 패턴을 학습할 수 있게 합니다.

SFT는 사전 학습 중에 획득한 기초 지식을 활용하면서 모델의 동작을 특정 요구사항에 맞게 조정할 수 있기 때문에 효과적입니다.

## Supervised Fine-Tuning을 사용해야 하는 경우

SFT 사용 결정은 주로 현재 모델의 능력과 특정 요구사항 사이의 간극에 따라 이루어집니다. SFT는 모델의 출력을 정밀하게 제어해야 하거나 전문화된 도메인에서 작업할 때 특히 가치가 있습니다.

예를 들어, 고객 서비스 애플리케이션을 개발하는 경우 모델이 일관되게 회사 지침을 따르고 기술적 문의를 표준화된 방식으로 처리하기를 원할 수 있습니다. 마찬가지로 의료나 법률 분야에서는 정확성과 도메인 특화 용어 준수가 매우 중요합니다. 이러한 경우 SFT는 모델의 응답을 전문적 기준과 도메인 전문성에 맞게 조정하는 데 도움이 됩니다.

## Fine-Tuning 프로세스

Supervised fine-tuning 프로세스는 작업별 데이터셋으로 모델의 가중치를 조정하는 것을 포함합니다.

먼저, 목표 작업을 대표하는 데이터셋을 준비하거나 선택해야 합니다. 이 데이터셋은 모델이 마주할 다양한 시나리오를 포함하는 예시들을 포함해야 합니다. 데이터의 품질이 중요한데 - 각 예시는 모델이 생성하기를 원하는 출력의 종류를 보여줘야 합니다. 다음으로 실제 fine-tuning 단계가 진행되는데, 여기서는 Hugging Face의 `transformers`와 `trl`과 같은 프레임워크를 사용하여 데이터셋으로 모델을 학습시킵니다.

전체 과정에서 지속적인 평가가 필수적입니다. 검증 세트에서 모델의 성능을 모니터링하여 일반적 능력을 잃지 않으면서 원하는 동작을 학습하고 있는지 확인해야 합니다. [module 4](../4_evaluation)에서 모델을 평가하는 방법을 다룰 것입니다.

## 선호도 조정에서 SFT의 역할

SFT는 언어 모델을 인간의 선호도에 맞게 조정하는 데 근본적인 역할을 합니다. Reinforcement Learning from Human Feedback (RLHF)와 Direct Preference Optimization (DPO)와 같은 기술들은 SFT를 통해 기본적인 작업 이해도를 형성한 후, 모델의 응답을 원하는 결과에 더 가깝게 조정합니다. 사전 학습된 모델들은 일반적인 언어 능력이 뛰어나더라도 항상 인간의 선호도에 맞는 출력을 생성하지는 않을 수 있습니다. SFT는 도메인 특화 데이터와 가이드를 도입함으로써 이러한 간극을 메우고, 모델이 인간의 기대에 더 가까운 응답을 생성할 수 있도록 합니다.

## Transformer Reinforcement Learning을 이용한 Supervised Fine-Tuning

Supervised Fine-Tuning을 위한 핵심 소프트웨어 패키지는 Transformer Reinforcement Learning (TRL)입니다. TRL은 강화학습(RL)을 사용하여 transformer 언어 모델을 학습시키는 데 사용되는 툴킷입니다.

Hugging Face Transformers 라이브러리 위에 구축된 TRL은 사용자가 사전 학습된 언어 모델을 직접 로드할 수 있게 하며 대부분의 decoder와 encoder-decoder 아키텍처를 지원합니다. 이 라이브러리는 supervised fine-tuning (SFT), reward modeling (RM), proximal policy optimization (PPO), Direct Preference Optimization (DPO)를 포함한 언어 모델링에 사용되는 주요 RL 프로세스를 용이하게 합니다. 우리는 이 리포지토리의 여러 모듈에서 TRL을 사용할 것입니다.

# 다음 단계

TRL을 사용한 SFT를 직접 경험해보려면 다음 튜토리얼을 시도해보세요:

⏭️ [Chat Templates 튜토리얼](./notebooks/chat_templates_example.ipynb)
⏭️ [Supervised Fine-Tuning 튜토리얼](./notebooks/sft_finetuning_example.ipynb)
