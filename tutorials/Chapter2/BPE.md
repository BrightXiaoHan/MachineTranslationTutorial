
# BPE分词原理

BPE(字节对)编码或二元编码是一种简单的数据压缩形式，其中最常见的一对连续字节数据被替换为该数据中不存在的字节。 后期使用时需要一个替换表来重建原始数据。将BPE算法原理应用在机器翻译分词部分源于这篇论文[Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/pdf/1508.07909)。后来各类NLP任务模型，如OpenAI GPT-2 与Facebook RoBERTa均采用此方法构建subword。作为传统的（以Moses为代表）基于空格、规则的分词方法的补充，使用BPE为代表的Subword分词算法主要解决了传统分词的以下几个痛点：
- 传统词表示方法无法很好的处理未知或罕见的词汇（OOV问题）
- 传统词tokenization方法不利于模型学习词缀之间的关系。如：模型学到的“old”, “older”, and “oldest”之间的关系无法泛化到“smart”, “smarter”, and “smartest”。
- Character embedding作为OOV的解决方法粒度太细
- Subword粒度在词与字符之间，能够较好的平衡OOV问题

## 算法原理
1. 准备足够大的训练语料，将语料进行预分词（中文可以使用jieba分词，拉丁语可以使用Moses，nltk等工具）。
2. 确定期望的subword词表大小
3. 将单词拆分为字符序列并在末尾添加后缀“ </ w>”，统计单词频率。 本阶段的subword的粒度是字符。 例如，“ low”的频率为5，那么我们将其改写为“ l o w </ w>”：5
4. 统计每一个连续字节对的出现频率，选择最高频者合并成新的subword
5. 重复第4步直到达到第2步设定的subword词表大小或下一个最高频的字节对出现频率为1

停止符"</w>"的意义在于表示subword是词后缀。举例来说："st"字词不加"</w>"可以出现在词首如"st ar"，加了"</w>"表明改字词位于词尾，如"wide st</w>"，二者意义截然不同。

每次合并后词表可能出现3种变化：

- +1，表明加入合并后的新字词，同时原来的2个子词还保留（2个字词不是完全同时连续出现）
- +0，表明加入合并后的新字词，同时原来的2个子词中一个保留，一个被消解（一个字词完全随着另一个字词的出现而紧跟着出现）
- -1，表明加入合并后的新字词，同时原来的2个子词都被消解（2个字词同时连续出现）

实际上，随着合并的次数增加，词表大小通常先增加后减小。

### 例子
输入：
```
{'l o w </w>': 5, 'l o w e r </w>': 2, 'n e w e s t </w>': 6, 'w i d e s t </w>': 3}
```
Iter 1, 最高频连续字节对"e"和"s"出现了6+3=9次，合并成"es"。输出：
```
{'l o w </w>': 5, 'l o w e r </w>': 2, 'n e w es t </w>': 6, 'w i d es t </w>': 3}
```
Iter 2, 最高频连续字节对"es"和"t"出现了6+3=9次, 合并成"est"。输出：
```
{'l o w </w>': 5, 'l o w e r </w>': 2, 'n e w est </w>': 6, 'w i d est </w>': 3}
```
Iter 3, 以此类推，最高频连续字节对为"est"和"<\/w>" 输出：
```
{'l o w </w>': 5, 'l o w e r </w>': 2, 'n e w est</w>': 6, 'w i d est</w>': 3}
```
……
Iter n, 继续迭代直到达到预设的subword词表大小或下一个最高频的字节对出现频率为1。

注意到输入BPE学习算法的数据是经过传统分词处理的，词内字符直接分割（Moses基于规则的分词，也可以加上Normalize，Truecase等操作），词尾字符增加</w>标志。所以这里也印证了前面说的BPE字词分割算法是作为传统预处理方法的一个补充。

### 代码实现
根据上文的算法原理，可以很容易地实现BPE学习算法


```python
# Code is adapt from original paper 
import re, collections

def get_stats(vocab):
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
            pairs[symbols[i],symbols[i+1]] += freq
    return pairs

def merge_vocab(pair, v_in):
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out

def build_vocab(text):
    counter = collections.defaultdict(int)
    for line in text:
        words = line.split()
        for word in words:
            counter[word] += 1
    return {" ".join(key) + " </w>": value for key, value in counter.items()}

test_corpus = '''Object raspberrypi functools dict kwargs . Gevent raspberrypi functools . Dunder raspberrypi decorator dict didn't lambda zip import pyramid, she lambda iterate ?
Kwargs raspberrypi diversity unit object gevent . Import fall integration decorator unit django yield functools twisted . Dunder integration decorator he she future . Python raspberrypi community pypy . Kwargs integration beautiful test reduce gil python closure . Gevent he integration generator fall test kwargs raise didn't visor he itertools ...
Reduce integration coroutine bdfl he python . Cython didn't integration while beautiful list python didn't nit !
Object fall diversity 2to3 dunder script . Python fall for : integration exception dict kwargs dunder pycon . Import raspberrypi beautiful test import six web . Future integration mercurial self script web . Return raspberrypi community test she stable .
Django raspberrypi mercurial unit import yield raspberrypi visual rocksdahouse . Dunder raspberrypi mercurial list reduce class test scipy helmet zip ?'''
        

vocab = build_vocab(test_corpus.split('\n'))
num_merges = 100
for i in range(num_merges):
    pairs = get_stats(vocab)
    if not pairs:
        break
    best = max(pairs, key=pairs.get)
    vocab = merge_vocab(best, vocab)

print(vocab)
```

    {'O bject</w>': 2, 'raspberrypi</w>': 10, 'functools</w>': 3, 'dict</w>': 3, 'kwargs</w>': 3, '.</w>': 14, 'G event</w>': 2, 'Dunder</w>': 3, 'decorator</w>': 3, "didn't</w>": 4, 'la m bd a </w>': 2, 'z ip </w>': 2, 'import</w>': 3, 'py ra m i d , </w>': 1, 'she</w>': 3, 'i te rat e</w>': 1, '? </w>': 2, 'K wargs</w>': 2, 'di v er s ity</w>': 2, 'unit</w>': 3, 'o bject</w>': 1, 'g event</w>': 1, 'I mport</w>': 2, 'fall</w>': 4, 'integration</w>': 8, 'd j a n g o </w>': 1, 'y i el d</w>': 2, 't w is te d</w>': 1, 'he</w>': 4, 'f ut ure</w>': 1, 'P ython</w>': 2, 'c o m m un ity</w>': 2, 'p yp y</w>': 1, 'beautifu l</w>': 3, 'test</w>': 5, 'r e d u c e</w>': 2, 'g i l</w>': 1, 'py thon</w>': 3, 'c l o s ure</w>': 1, 'g en e rator</w>': 1, 'ra is e</w>': 1, 'v is or</w>': 1, 'it er tools</w>': 1, '. . .</w>': 1, 'R e d u c e</w>': 1, 'c or o ut in e</w>': 1, 'bd f l</w>': 1, 'C ython</w>': 1, 'w h i l e</w>': 1, 'l i st</w>': 2, 'n it</w>': 1, '! </w>': 1, '2 to 3 </w>': 1, 'd under</w>': 2, 's c r ip t</w>': 2, 'f or</w>': 1, ': </w>': 1, 'e x c e p t ion</w>': 1, 'py c on</w>': 1, 's i x </w>': 1, 'w e b </w>': 2, 'F ut ure</w>': 1, 'm er c ur i al</w>': 3, 's el f </w>': 1, 'R e t ur n</w>': 1, 's t a b l e</w>': 1, 'D j a n g o </w>': 1, 'v is u al</w>': 1, 'r o c k s d a h o u s e</w>': 1, 'c la s s</w>': 1, 's c i py </w>': 1, 'h el m e t</w>': 1}


经过100轮的迭代，可以看到一些出现频率较高的子词已经被合并在一起了。但是这样的词表没有办法直接使用，需要将它们合并成`字词：频率`对的形式。


```python
def build_dict_from_vocab(vocab):

    subword_dict = collections.defaultdict(int)

    for key, value in vocab.items():
        for subword in key.split():
            subword_dict[subword] += value
    return subword_dict

subword_dict = build_dict_from_vocab(vocab)
sorted_subword_dict = sorted(subword_dict.items(), key=lambda x: -x[1])
 
# 这里只输出前二十个
for subword, num in sorted_subword_dict[:20]:
    print(subword.ljust(20), num)
```

    </w>                 17
    c                    17
    .</w>                15
    i                    13
    s                    12
    m                    11
    raspberrypi</w>      10
    e                    10
    e</w>                9
    o                    9
    d                    8
    integration</w>      8
    a                    6
    py                   6
    er                   6
    g                    5
    l</w>                5
    test</w>             5
    r                    5
    u                    5


可以看到一些常见的字词组合如`.</w>`,`integration`等已经被学习到了。由于语料比较小，有些字词还是以单个英文字母的形式存在，所以需要训练语料量足够大，才能学习到足够好的subword词典。

## 编码和解码
在之前的算法中，我们已经得到了subword的词表，对该词表按照子词长度由大到小排序。编码时，对于每个单词，遍历排好序的子词词表寻找是否有token是当前单词的子字符串，如果有，则该token是表示单词的tokens之一。

我们从最长的token迭代到最短的token，尝试将每个单词中的子字符串替换为token。 最终，我们将迭代所有tokens，并将所有子字符串替换为tokens。 如果仍然有子字符串没被替换但所有token都已迭代完毕，则将剩余的子词替换为特殊token，如`<unk>`。

### 例子

为了简单起见，我们给定单词序列
```
[“the</w>”, “highest</w>”, “mountain</w>”, “largest</w>”]
```
假设已有排好序的subword词表
```
[“err</w>”, “tain</w>”, “moun”, “est</w>”, “high”, “the</w>”, “a</w>”]
```
迭代结果
```
"the</w>" -> ["the</w>"]
"highest</w>" -> ["high", "est</w>"]
"mountain</w>" -> ["moun", "tain</w>"]
"largest</w>" -> ["<unk>", "est"]
```

### 代码实现



```python
subword_dict = {"err</w>": 2, "tain</w>":3, "moun":3, "est</w>":4, "high":2, "the</w>":5, "and</w>":10, "l":1, "g":1, "a":1, "r":1}
ngram_max = max(len(x) for x in subword_dict)
UNK = "<unk>"

def subword_tokenize(word):
    word += "</w>"
    end_idx = min([len(word), ngram_max])
    sw_tokens = []
    start_idx = 0

    while start_idx < len(word):
        subword = word[start_idx:end_idx]
        if subword in subword_dict:
            sw_tokens.append(subword)
            start_idx = end_idx
            end_idx = min([len(word), start_idx + ngram_max])
        elif subword == "</w>":
            break
        elif len(subword) == 1:
            sw_tokens.append(UNK)
            start_idx = end_idx
            end_idx = min([len(word), start_idx + ngram_max])
        else:
            end_idx -= 1

    return sw_tokens

def tokenize(sentence):
    """给定预训练的词表和待分词的句子（默认已经通过nltk或者moses等工具进行过预分词），输出基于bpe的分词结果
    
    Args:
        sentence (str): 待分词的句子
        
    Return:
        list: 字词列表
    """
    tokens = []
    
    for word in sentence.split():
        tokens.extend(subword_tokenize(word))
        
    return tokens


tokens = tokenize("the highest and largest mountain 好")
print(tokens)
```

    ['the</w>', 'high', 'est</w>', 'and</w>', 'l', 'a', 'r', 'g', 'est</w>', 'moun', 'tain</w>', '<unk>']


解码的代码就比较简单了，以`</w>`为标志做词的划分，其余相邻的词直接进行合并。


```python
def detokenize(tokens):
    return "".join(tokens).replace("</w>", " ").replace("<unk>", "")

print(detokenize(tokens))
```

    the highest and largest mountain 

