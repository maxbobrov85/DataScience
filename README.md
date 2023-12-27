# Пространство действий     env/bpp0/space.py
### Класс Rotate
    Именованное определение поворота коробок
### Класс Box
    Определение коробок
    dx,dy,dz - размеры
    x,y,z - координаты нижнего левого дальнего угла.
    Метод rotate - определение поворота коробки
### Класс Space
    Описание контейнера
        plain_size размеры
        plain проекция текущего расположения коробок
        boxes расположенные коробки
        flags признак поворота коробок
        height высота
        Методы:
            update_height_graph обновляет plain после размещения очередной коробки в контейнере
            check_box - проверка возможности размещения очередной коробки на данную позицию.
                        Требование что бы коробка опиралась как минимум на три угла, при этом опираться по площади должна в случае опоры на 3 угла: на 85% поверхности; 4 угла - 50%.
            get_ratio - метрика заполнености контейнера
            idx_to_position - перевод возвращаемого значения положения коробки модели в координаты размещения
            position_to_index - перевод координат в значение для модели
            drop_box - помещение коробки в контейнер
# Модель model.py
Базово реализуем алгоритм предложенные в статье [Jingwei Zhang, Bin Zi, Xiaoyu Ge, Attend2Pack: Bin Packing through Deep Reinforcement Learning with Attention](https://arxiv.org/abs/2107.04333)
### Класс Embedding 
Представление коробок в наборе. В начале каждого эпизода размерности каждого ящика на входе сперва преобразуются с помощью линейного слоя:
```math
\begin{align}
\bar{{\mathsf{b}}}^n=\mathrm{Linear} (l^n,w^n,h^n)
\end{align}
```
### Класс Encoder (состоит из EncoderMultiHeadAttention (2) и FeedForward(3))
Множество из $N$ таких размещений $\bar{{\mathsf{b}}}$
($\lvert \bar{{\mathsf{b}}} \rvert=d$) затем проходит через несколько  multi-head (с $M$ heads) self-attention слоев , каждый из которых содержит следующие операции.
```math
\tilde{{\mathsf{b}}}^n = \bar{{\mathsf{b}}}^n + \mathrm{MHA}\big(\mathrm{LN}\big(\bar{{\mathsf{b}}}^1, \bar{{\mathsf{b}}}^2,\ldots,\bar{{\mathsf{b}}}^N \big)\big),
```
```math
{\mathsf{b}}^n =\tilde{{\mathsf{b}}}^n + \mathrm{MLP}\big(\mathrm{LN}\big(\tilde{{\mathsf{b}}}^n\big) \big).
```
где $\mathrm{MHA}$ обозначает multi-head attention слой , $\mathrm{LN}$ обозначает слой нормализации , а $\mathrm{MLP}$ обозначает полносвязный слой с функцией активации $\mathrm{ReLU}$. 

### Класс FrontierEmbbeding
Представление предыдущей и текущей ситуации в контейнере.
Для компенсации возможных наложений,
на каждом шаге времени $t$
мы объединяем последние две границы
${{\mathsf{F}}{t-1}, {\mathsf{F}}{t}}$
(${\mathsf{F}}_{t}$
обозначает границу перед упаковкой коробки
$(l^{s_t}, w^{s_t}, h^{s_t})$
в контейнер)
и передаем их через сверточную сеть
для получения векторного представления границы
${\mathsf{f}_t}$.
### Класс PlacementDecoder
Декодирование последовательности выбора положения коробки. Простой линейный слой.

## Класс PointerNet 
Собрали все вместе и добавляем технику glympsy attention 
### Функция sequentional_embedding

Определяет функцию кодирования $f_{\mathcal{I}}^{\mathrm{s}}$ (4) для  политики выбора следующей (sequence) и $f_{\mathcal{I}}^{\mathrm{p}}$ (5) размещения (placement) следующим образом:
```math
\bar{{\mathsf{q}}}^{\mathrm{s}}_t=f^{\mathrm{s}}_{\mathcal{I}}(s_{1:t-1}, p_{1:t-1}; {\theta}^{\mathrm{e}})=\big\langle\langle\mathcal{B}\setminus{{\mathsf{b}}^{s_{1:t-1}} }\rangle,{\mathsf{f}}_{t}\big\rangle,
```
```math
{\mathsf{q}}^{\mathrm{p}}_t=f^{\mathrm{p}}_{\mathcal{I}}(s_{1:t}, p_{1:t-1}; {\theta}^{\mathrm{e}})=\big\langle{\mathsf{b}}^{s_t}, \langle\mathcal{B}\setminus{{\mathsf{b}}^{s_{1:t}} }\rangle,{\mathsf{f}}_{t}\big\rangle.
```
где
$\langle\rangle$
представляет операцию, которая принимает на вход
набор $d$-мерных векторов
и возвращает их средний вектор,
также $d$-мерный.
Эта функция кодирования генерирует
вектор запроса
$\bar{{\mathsf{q}}}^{\mathrm{s}}_t$
для декодирования последовательности размещения.
### Функция seqPolicy
Реализует декодирование последовательности выбора следующей коробки. В соответствии с [Bello, I., Pham, H., Le, Q. V., Norouzi, M., and Bengio, S. Neural combinatorial optimization with reinforcement
learning.](https://arXiv:1611.09940),
мы добавляем операцию обзора
с использованием multi-head 
($M$ heads)
внимания
[Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones,
L., Gomez, A. N., Kaiser, Ł., and Polosukhin, I. Attention
is all you need.](https://arxiv.org/abs/1706.03762)
перед расчетом вероятностей стратегии,
что, как показано, приводит к повышению производительности.
В начале каждого эпизода,
из каждого представления размещения
${\mathsf{b}}^n$
получаются $M$ множеств фиксированных контекстных векторов
с помощью обучаемых линейных проекций.
Где $h=\frac{d}{M}$,
контекстные векторы
$\bar{{\mathsf{k}}}^{n,m}$ (обзорный ключ размером $h$) и
$\bar{{\mathsf{v}}}^{n,m}$ (обзорное значение размером $h$)
для $m^{\mathrm{th}}$ head,
а также
${\mathsf{k}}^{n}$ (ключ логита размером $d$),
общий для всех $M$ head,
получаются с помощью
```math
    \bar{{\mathsf{k}}}^{n,m}
=
    {\mathsf{W}}^{\bar{{\mathsf{k}}},m}
    {\mathsf{b}}^n,
\hspace{5pt}
% \\
    \bar{{\mathsf{v}}}^{n,m}
=
    {\mathsf{W}}^{\bar{{\mathsf{v}}},m}
    {\mathsf{b}}^n,
\hspace{5pt}
    % \\
    {\mathsf{k}}^{n}
=
    {\mathsf{W}}^{{\mathsf{k}}}
    {\mathsf{b}}^n,
```
где ${\mathsf{W}}^{\bar{{\mathsf{k}}},m}, {\mathsf{W}}^{\bar{{\mathsf{v}}},m} \in\mathbb{R}^{h\times d}$ и ${\mathsf{W}}^{{\mathsf{k}}}in\mathbb{R}^{d\times d}$. Вектор запроса последовательности $\bar{{\mathsf{q}}}^{\mathrm{s}}_t$ is split into $M$ векторы запроса для обзора, каждый размерностью
$\lvert \bar{{\mathsf{q}}}^{\mathrm{s},m}_t \rvert=h$.
Совмещенный вектор 
$\bar{{\mathsf{c}}}^m_t$ 
($\lvert \bar{{\mathsf{c}}}^m_t \rvert=N$) 
вычисляется следующим образом
```math
    \bar{{\mathsf{c}}}^m_{t,n}
=
    \begin{cases}
        \frac{
                {\bar{{\mathsf{q}}}_t^{{\mathrm{s},m}^{\top}}} 
                \bar{{\mathsf{k}}}^{n,m}
             }
             {\sqrt{h}} 
& 
        \text{if} \hspace{2pt} n\notin \{s_{1:t-1}\},
\\
        -\infty 
& 
        \text{иначе}.
    \end{cases}

```
Далее получаем обновленный запрос последовательности для $m^{\mathrm{th}}$ head взвешенной суммой векторов значений обзора
```math
    {\mathsf{q}}^{\mathrm{s},m}_t =
    \sum_{n=1}^N
    \mathrm{softmax} (\bar{{\mathsf{c}}}^m_{t})_{n}
    \cdot
    \bar{{\mathsf{v}}}^{n,m}.
```
Объединяя обновленные векторы запроса последовательности из всех $M$ heads, мы получаем

${\mathsf{q}}^{\mathrm{s}}_t$
($\lvert {\mathsf{q}}^{\mathrm{s}}_t \rvert=d$).
Применив еще одну линейную проекцию, мы получаем

${\mathsf{W}}^{{\mathsf{q}}}
\in
\mathbb{R}^{d\times d}$,
обновленный совмещенный вектор 
${\mathsf{c}}_t$ ($\lvert {\mathsf{c}}_t \rvert=N$) 
вычисляется как
```math
    {\mathsf{c}}_{t,n}
=
    \begin{cases}
        C \cdot \mathrm{tanh} 
        \Big( 
        \frac{
                ({\mathsf{W}}^{{\mathsf{q}}}
                 {\mathsf{q}}^{\mathrm{s}}_t)^{\top}
                {\mathsf{k}}^{n}
             }
             {\sqrt{d}} 
        \Big)
        \text{if} \hspace{2pt} n\notin \{s_{1:t-1}\},
\\
        -\infty 
        \text{иначе},
    \end{cases}

```
где логиты ограничены значениями между
$[-C,C]$ следуя
[Kool, W., van Hoof, H., and Welling, M. Attention, learn to solve routing problems!](https://arxiv.org/abs/1803.08475)
так как это может способствовать исследованию и привести к повышению производительности.
Наконец, стратегия последовательности получена с помощью
$$

    \pi^{\mathrm{s}}
        \big(
            \cdot 
            \big\vert 
            f^{\mathrm{s}}_{\mathcal{I}}(s_{1:t-1}, p_{1:t-1}; {\theta}^{\mathrm{e}}); 
            {\theta}^{\mathrm{s}}
        \big)
=
    \mathrm{softmax}({\mathsf{c}}_t).


$$
### Функция sample_action
Реализует выбор коробки и ее положение случайным образом по распределению  $f_{\mathcal{I}}^{\mathrm{s}}$ (4)  и $f_{\mathcal{I}}^{\mathrm{p}}$ (5)

# Gradient Policy utils.py
### Функция compute_reinforce_with_baseline_loss
реализует градиентную политику описанную {Williams, R. J. Simple statistical gradient-following algorithms for connectionist reinforcement learning.}. При заданном входном множестве $\mathcal{I}$, как агент последовательности $\pi^{\mathrm{s}}$ и агент размещения $\pi^{\mathrm{p}}$ сотрудничают, чтобы получить конфигурацию решения $\mathcal{C}{\pi}={s_1, p_1, s_2, p_2, \ldots, s_N, p_N}$, работая совместно на протяжении $N$ временных шагов ($\pi$ обозначает совокупность $\pi^{\mathrm{s}}$ и $\pi^{\mathrm{p}}$). Чтобы обучить эту систему таким образом, чтобы $\pi^{\mathrm{s}}$ и $\pi^{\mathrm{p}}$ могли сотрудничать для максимизации окончательной выгоды $r{\mathrm{u}}\in[0,1]$, что эквивалентно минимизации стоимости $c(\mathcal{C}{\pi})=1-r{\mathrm{u}}(\mathcal{C}_{\pi})$, определим общую функцию потерь как ожидаемую стоимость конфигураций, сгенерированных при использовании $\pi$:
$$

    \mathcal{J}({\theta} \vert \mathcal{I})
=
    \mathbb{E}_{\mathcal{C}_{\pi}\sim\pi_{{\theta}}}
    \left[
        c(\mathcal{C}_{\pi} \vert \mathcal{I})
    \right],

$$
оптимизируемая следующей $\mathrm{REINFORCE}$ градиентной оценкой 
$$

    &\nabla_{{\theta}}
    \mathcal{J}({\theta} \vert \mathcal{I})
=
    \mathbb{E}_{\mathcal{C}_{\pi}\sim\pi_{{\theta}}}
    \Big[
        \Big(
            c(\mathcal{C}_{\pi} \vert \mathcal{I})
            - 
            b(\mathcal{I})
        \Big)
    \cdot
\notag\\&
    \sum_{t=1}^{N}
        \Big(
       
        \nabla_{{\theta}^{\mathrm{e},\mathrm{s}}}
        \log \pi^{\mathrm{s}} 
        (
            s_t; 
            
            {\theta}^{\mathrm{e},\mathrm{s}}
            
        )
        +    
        
        \nabla_{{\theta}^{\mathrm{e},\mathrm{p}}}
        \log \pi^{\mathrm{p}}
        (
            p_t; 
            
            {\theta}^{\mathrm{e},\mathrm{p}}
            
        )
        \Big)
    \Big].


$$
Для базовой функции (baseline function) $b(\mathcal{I})$ в указанном уравнении мы используем жадную базовую политику (greedy rollout baseline), предложенную в работе [Kool, W., van Hoof, H., and Welling, M. Attention, learn to solve routing problems!](https://arxiv.org/abs/1803.08475)


