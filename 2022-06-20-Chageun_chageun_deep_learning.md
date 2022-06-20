---
layout : single
title : Chageun Chageun Deep Learning
---

# 01 딥러닝으로의 출발
## 딥러닝 이해하기
like 피처 벡터 in 학습데이터 &rarr; 가중치 &rarr; 첫눈에 반했다는 느낌  
> 최적해 (Optimal Solution) 를 찾았다!

지금껏 만난 사람들을 기준으로 early stopping 후 생성된 모델 &rarr; 어느 정도 기준에 맞는다면 최적해로 판단해줘 &rarr; but 더 좋은 사람을 만날 가능성도 있다!  
그래서 early stopping 을 통해 찾은 답을
> 부분 최적해 (Sub-optimal Solution) 이라고 한다

이 부분 최적해는 다양하게 존재할 수 있다!

## DNA
: Data + Network + AI

## 딥러닝은 왜 뛰어날까?
과거에 경험한 것과 같은 상황이 반복 발생한 경우 &rarr; 과거 검색 후 똑같이 대응하면 됨  
but 새롭게 발생한 상황이라면 &rarr; 데이터 학습에 의한 추론 능력이 필요  

선형 데이터 &rarr; 머신러닝  
비선형 데이터 &rarr; 머신러닝  
비선형의 정도가 심한 데이터 (*i.e.* 소용돌이) &rarr; **?**

비선형의 정도가 심해지는 것 :  
> 차원 (Dimension) 이 증가한다  

기존의 방법들 (머신러닝) 은 그 성능이 급격히 저하된다 (차원의 저주, Curse of Dimensionality)  
> 딥러닝은 차원 축소를 잘하는 동시에 비선형 데이터를 잘 흉내낼 수 있는 대표적 알고리즘

## 딥러닝으로의 출발
데이터 &rarr;   
전처리 &rarr;   
함수 = 로직 = 알고리즘 = 모델 &rarr;  
결과 = 출력  

학습 (learning) : 분명히 존재하는 가장 정확한 함수 $f^*$ 와 가장 유사한 $f$ 를 찾아가는 과정  

## 데이터와 파라미터
MLE (Maximum Likelihood Estimation) : 수학적 과정을 통해 어떤 분포로 추정하는 것  

평균 $\mu$  
표준편차 $\sigma$  
정규분포
$$f_\mu, _\sigma(x) = \frac{1}{\sigma\sqrt{2\pi}}e^{-\frac{(x-\mu)^2}{2\sigma^2}} $$  

의 우변에서 $x$ 를 제외하고  
자연상수 $e$ = 2.72  
원주율 $\pi$ = 3.14   
이므로 정규분포의 형태를 결정할 수 있는 것은 평균 $\mu$,  표준편차 $\sigma$  
즉 특징을 결정하는 파라미터들인 것!

## 인공신경망
**UAT (Universal Approximation Theorem)** : 조지 시벤코(Cybenko) 가 1989년 발표한 정리
> "인공신경망 구조가 어떠한 함수라도 근사할 수 있다!"   
> = 세상의 거의 모든 함수를 따라 할 수 있다

`# 테일러 급수에서 다항함수를 가지고 / 퓨리에 급수에서 삼각함수를 이용해 함수를 묘사하는 것과 같은 의미`
$$G(x, w, \alpha, \theta) = \sum_{i=1}^{N}\alpha_i\sigma(w_i^Tx+\theta_i)$$
여기서  
$\sigma$ 는 시그모이드 함수  
$w_i$, $x_i$는 n차원 실수 벡터  
$\theta_i$, $\alpha_i$ 는 실수  

## 병렬처리와 딥러닝
- computer 의 구동 과정  
  1. 입력장치 (키보드, 마우스 등) 를 이용해 컴퓨터에 명령
  2. 명령 &rarr; CPU
      - ALU (Arithmetic Logic Unit) : 산술 논리 담당
      - 레지스터 : 명령어를 저장하는 고속 저장장치
      - CU (Control Unit) : CPU 내부 제어
  3. 필요한 경우 보조기억장치 (HDD, SSD) 에 들어있는 프로그램을 주기억장치 (RAM) 에 적재
  4. 이 때 적재된 프로그램 = 프로세스 (process)
<br><br>
- CPU 는 입력된 순서대로 작업을 처리하는 순차형 처리 방식에 특화되어 있다
- 결과적으로 CPU 에는 많은 수의 ALU 가 장착되지 않는다
- 대신 빠르게 연산을 처리하기 위해 **캐시 메모리** 가 차지하는 비중이 더 크다  
- (최근에 사용한 명령어나 데이터를 보관했다가 ALU 의 요청이 있으면 메인 메모리까지 가지 않고 빠르게 처리할 수 있게 함)

|CPU|GPU|
|-|-|
|ALU 4개|ALU 128개|
|control<br>cache<br><br>DRAM|control<br>cache<br>(but 비중이 적음)<br>DRAM|
|1대의 비행기|128대의 기차|

- 그러나 GPU 가 병렬처리를 한다고 항상 빠른 것은 아니다
- GPU 가 많으면 어떤 작업은 놀고 있는 ALU 가 많아지면서 성능이 급격하게 떨어질 수 있다
<br><br>
- 2012년 구글의 '구글 브레인'
- CPU 서버 1000대 병렬 연결
- 50억원
<br><br>
- Nvidia
- GPU 가속 서버 3대로 구글을 이김
- 3천 3백만원
- 2006년 CUDA (Compute Unified Device Architecture) 병렬 플랫폼 개발