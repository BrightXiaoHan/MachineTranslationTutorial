
# Truecase

在英语等一些大小写敏感的语言中，一些专有名词和有特殊用法的单词，以及每个句子的首字母都需要进行大写。此外，训练数据中也会包括一些大小写错误的用法。这导致许多单词由于大小写的区分存在多种形式。一种简单的做法是将数据全部进行小写化，这样可以使所有的单词进行统一，大大提升模型预测的准确性。然而，用小写化数据训练的模型翻译结果也都是小写的，需要额外的还原模型对结果进行处理。

 现在更常用的做法是保留句子中每个单词的正确大小写形式。但是对于句子的首字母，需将其转换成这个单词最常见的形式，如下表所示。
 
 What is the WTO ? 
 
 - Lowercase: what is the wto ?
 - Truecase: what is the WTO ? 


通过这种方式，训练数据中只包含单词的正确大小写形式，大写单词只存在于一些专有名词或者有特殊用法的单词中，在一定程度上减小了词表大小，同时，也去除了一部分数据中由于错误大小写形式所产生的噪音。在翻译结束后，对首字母进行大写就能得到大小写合理的翻译结果。另外，中文存在简繁体两种形式的汉字，训练数据中可能会同时包含这两种形式。因此通常也会考虑把繁体中文转化为简体中文，以统一汉字的编码。
 

本节主要介绍如何训练Truecase模型，对训练数据进行Truecase处理，以及对Truecase数据进行还原（Detruecase）。

## 训练Truecase模型
由于Truecase是针对某一种语言的，并不要求一定要使用双语语料进行训练，还可以利用获取成本较低的单语语料进行训练。我们首先准备一个小的数据集来做实验。由于Truecase是以词为单位进行学习训练的，所以在做Truecase之前，先要对语料进行分词处理。这里使用sacremoses中的分词脚本进行分词。具体的分词流程与原理，在EnglishTokenizer章节中进行了详细的介绍。


```python
# 安装sacremoses
!pip -q install -i https://pypi.douban.com/simple sacremoses 
# 获取训练数据
!wget -q https://gist.githubusercontent.com/alvations/6e878bab0eda2624167aa7ec13fc3e94/raw/4fb3bac1da1ba7a172ff1936e96bee3bc8892931/big.txt
# 对数据进行分词处理
!sacremoses -l en -j 4 tokenize  < big.txt > big.txt.tok
```

    [33mWARNING: You are using pip version 20.2.4; however, version 20.3.1 is available.
    You should consider upgrading via the '/root/Softwares/anaconda3/bin/python -m pip install --upgrade pip' command.[0m
    100%|█████████████████████████████████| 128457/128457 [00:15<00:00, 8344.83it/s]



```python
!head big.txt.tok
```

    The Project Gutenberg EBook of The Adventures of Sherlock Holmes
    by Sir Arthur Conan Doyle
    ( # 15 in our series by Sir Arthur Conan Doyle )
    
    Copyright laws are changing all over the world . Be sure to check the
    copyright laws for your country before downloading or redistributing
    this or any other Project Gutenberg eBook .
    
    This header should be the first thing seen when viewing this Project
    Gutenberg file . Please do not remove it . Do not change or edit the


训练Truecase模型的原理其实非常简单，我们只需要统计每个单词不同形态下的词频。比如单词 “internet”，在我们的训练语料中有三种形态，分别是“internet”，“Internet”，“INTERNET”，这三种形态在训练语料中出现的频率分别是1，100，2次。当模型从训练数据中学习到这种分布特征之后，在平行语料预处理、后处理阶段，就能对不同Case的“internet”进行处理（具体处理方法细节后面会讲）。

首先我们编写统计一句话中每个词不同形式的词频的代码。



```python
import re

# 如果遇到这些词，这些词不能作为句子的开头，通常下一个词才是。如“( Additional editing by Jose Menendez )”
DELAYED_SENT_START = {
    "(",
    "[",
    '"',
    "'",
    "&apos;",
    "&quot;",
    "&#91;",
    "&#93;",
}

# 如果遇到这些词意味着当前句子结束，下一个单词可能是句子的开头。
SENT_END = {".", ":", "?", "!"}

# 该正则用于跳过不包含大写字母、小写字母和标题字母的词。如纯数字，纯符号 “( # 15 in our series by Sir Arthur Conan Doyle )”
Lowercase_Letter = open("assets/Lowercase_Letter.txt").read()
Uppercase_Letter = open("assets/Uppercase_Letter.txt").read()
Titlecase_Letter = open("assets/Titlecase_Letter.txt").read()

SKIP_LETTERS_REGEX = re.compile(
    u"[{}{}{}]".format(
        Lowercase_Letter, Uppercase_Letter, Titlecase_Letter
    )
)

def learn_truecase_weights(tokens):
    """
    tokens: 句子的分词结果.
    """
    # 下一个词是否是句首单词的标记，如果是句首单词可能不计入统计
    is_first_word = True
    truecase_weights = []
    for i, token in enumerate(tokens):
        # 跳过xml标记中的词。这些词在分词时往往是一个整体，里面的词的Case与句首词语Case一样没有统计意义。
        if re.search(r"(<\S[^>]*>)", token):
            continue
        # 如果遇到这些词，这些词不能作为句子的开头，通常下一个词才是。如“( Additional editing by Jose Menendez )”
        elif token in DELAYED_SENT_START:
            continue

        # 如果遇到这些词意味着当前句子结束，下一个单词可能是句子的开头。重置 is_first_word
        if not is_first_word and token in SENT_END:
            is_first_word = True
            continue
        # 跳过不需要进行大小写统计的词，如数字、符号或者他们的组合
        if not SKIP_LETTERS_REGEX.search(token):
            is_first_word = False
            continue

        # 将当前词的统计结果加入到truecase_weights中。如 (lowercasetoken, LowerCaseToken, 1)
        current_word_weight = 0
        if not is_first_word:
            current_word_weight = 1

        is_first_word = False

        if current_word_weight > 0:
            truecase_weights.append((token.lower(), token, current_word_weight))
    return truecase_weights

example = "Copyright laws are changing all over the world . Be sure to check the copyright laws for your country before downloading or redistributing this or any other Project Gutenberg eBook ."
learn_truecase_weights(example.split())
```




    [('laws', 'laws', 1),
     ('are', 'are', 1),
     ('changing', 'changing', 1),
     ('all', 'all', 1),
     ('over', 'over', 1),
     ('the', 'the', 1),
     ('world', 'world', 1),
     ('sure', 'sure', 1),
     ('to', 'to', 1),
     ('check', 'check', 1),
     ('the', 'the', 1),
     ('copyright', 'copyright', 1),
     ('laws', 'laws', 1),
     ('for', 'for', 1),
     ('your', 'your', 1),
     ('country', 'country', 1),
     ('before', 'before', 1),
     ('downloading', 'downloading', 1),
     ('or', 'or', 1),
     ('redistributing', 'redistributing', 1),
     ('this', 'this', 1),
     ('or', 'or', 1),
     ('any', 'any', 1),
     ('other', 'other', 1),
     ('project', 'Project', 1),
     ('gutenberg', 'Gutenberg', 1),
     ('ebook', 'eBook', 1)]



接下来，我们对训练语料中的每一句话的词频做统计，并将统计结果合并。


```python
# 首先读取训练数据
with open("big.txt.tok", 'r') as f:
    corpus = f.readlines()

from collections import defaultdict, Counter
# 数据结构用于统计每个单词不同词频
casing = defaultdict(Counter)

token_weights = []
for line in corpus:
    token_weights.extend(learn_truecase_weights(line.split()))

for lowercase_token, surface_token, weight in token_weights:
    casing[lowercase_token][surface_token] += weight

# 将统计结果分成best，known两部分。best表示统计频数最高的大小写形式，know表示其他的大小写形式
best = {}
# 此处为了保证know中的每个元素可以通过字典的形式访问，所以这里用一个Counter，每个元素的值默认为1
known = Counter()

for token_lower in casing:
    tokens = casing[token_lower].most_common()
    best[token_lower] = tokens[0][0]
    for token, count in tokens[1:]:
        known[token] += 1
model = {"best": best, "known": known, "casing": casing}    
```

在进行Truecase操作前，输入的文本通常是经过分词处理后的文本，首先将他们以空格为分隔符分成单词（如果文本中有xml格式的文本，也将其包裹的单词分割开来。）


```python
def split_xml(line):
    """
    将文本以空格为分隔字符分开，如果文本中包含xml格式的文本，也将他们分开。
    如 hello <heading>Reminder</heading> 会将它分割成
    ['hello', '<heading>', 'Reminder', '</heading>']
    """
    line = line.strip()
    tokens = []
    while line:
        # Assumes that xml tag is always separated by space.
        has_xml = re.search(r"^\s*(<\S[^>]*>)(.*)$", line)
        # non-XML test.
        is_non_xml = re.search(r"^\s*([^\s<>]+)(.*)$", line)
        # '<' or '>' occurs in word, but it's not an XML tag
        xml_cognates = re.search(r"^\s*(\S+)(.*)$", line)
        if has_xml:
            potential_xml, line_next = has_xml.groups()
            # exception for factor that is an XML tag
            if (
                re.search(r"^\S", line)
                and len(tokens) > 0
                and re.search(r"\|$", tokens[-1])
            ):
                tokens[-1] += potential_xml
                # If it's a token with factors, join with the previous token.
                is_factor = re.search(r"^(\|+)(.*)$", line_next)
                if is_factor:
                    tokens[-1] += is_factor.group(1)
                    line_next = is_factor.group(2)
            else:
                tokens.append(
                    potential_xml + " "
                )  # Token hack, unique to sacremoses.
            line = line_next

        elif is_non_xml:
            tokens.append(is_non_xml.group(1))  # Token hack, unique to sacremoses.
            line = is_non_xml.group(2)
        elif xml_cognates:
            tokens.append(
                xml_cognates.group(1)
            )  # Token hack, unique to sacremoses.
            line = xml_cognates.group(2)
        else:
            raise Exception("ERROR: huh? {}".format(line))
        tokens[-1] = tokens[-1].strip()  # Token hack, unique to sacremoses.
    return tokens

text = "hello <heading>Reminder</heading>"
split_xml(text)
```




    ['hello', '<heading>', 'Reminder', '</heading>']



有了模型和输入文本之后，我们就可以使用模型对文本进行Truecase处理。


```python
def truecase(text, model, return_str=False, use_known=False):
    """
    对一句话或者一段文本进行Truecase操作
    
    Args:
        text (str): 输入文本（已经经过分词处理）
        model (dict): 从训练数据中学习到的case的统计数据
        return_str (bool, optional): 以str的形式返回还是以List[str]的形式返回. Defaults to True.
        use_known (bool, optional): 当该参数为True时，当某个词不是句首单词，并且是在训练数据中出现过的大小写形式，则保留原大小写形式不变。
                                    当该参数为False时，优先使用该词最常见的大小写形式
    """
    # 记录当前单词是否应为句首单词
    is_first_word = True
    truecased_tokens = []
    tokens = split_xml(text)

    for i, token in enumerate(tokens):
        # 这里以 ”|“ 符号开头的单词不做处理。注：这里为什么要对这个符号做特殊处理还不太清除
        if token == "|" or token.startswith("|"):
            truecased_tokens.append(token)
            continue
        
        # 处理这种情况  ”hello|thankyou“ -> token="hello", other_fectors="|thankyou"是处理词中有”|符号的情况“
        token, other_factors = re.search(r"^([^\|]+)(.*)", token).groups()

        # 最常见的（训练中频数最高的）单词大小写形式
        best_case = model["best"].get(token.lower(), None)
        # 其他的单词大小写形式
        known_case = model["known"].get(token, None)
  
        if is_first_word and best_case:  # 句首单词采用最常见的大小写形式
            token = best_case
        elif known_case and use_known:  # 在训练集中出现过的并且use_known=True大小写形式保持不变
            token = token
        elif (
            best_case
        ):  # 如果匹配到best_case使用最常见的大小写形式
            token = best_case
        # 否则是没有见过的单词，大小写形式也保持不变
        
        # 处理之前以”|“将词分开的情况，将他们重新拼接在一起
        token = token + other_factors
        # Adds the truecased
        truecased_tokens.append(token)

        # 遇见句末标点重置句首标志
        is_first_word = token in SENT_END
        
        # 延迟将句首标志置为False
        if token in DELAYED_SENT_START:
            is_first_word = False

    # 根据return_str参数判断是以词的形式返回还是以字符串的形式返回
    return " ".join(truecased_tokens) if return_str else truecased_tokens
```

找一段文本来试一下效果


```python
input_str = "You can also find out about how to make a donation to Project Gutenberg, and how to get involved."
output_str = truecase(input_str, model, return_str=True)
print(output_str)
```

    you can also find out about how to make a donation to Project Gutenberg, and how to get involved.


可以看到，首字母You变成了小写，人名Project Gutenberg还保留了原来的形式。
