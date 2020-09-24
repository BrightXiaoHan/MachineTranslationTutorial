
# 英文分词
分词是数据预处理的第一步。。对于像中文这样没有单词边界的语言，分词的策略通常比较复杂。现在常用的一些中文分词工具有 NLTK、jieba等。而像英文这种有单词边界的语言，分词要简单许多，比如，Moses 工具包就有可以处理绝大多数拉丁语系语言的分词脚本。

本章节就以[sacremoses](https://github.com/alvations/sacremoses)为例，讲解英文的分词过程。

目录：
1. 替换空白字符
2. 去掉句子开头和结尾的空白字符
3. 将常见标点、乱码等符号与词语分开
4. 分割逗号`,
5. 分割句号`,`
6. 处理`'`号缩写
7. 可选处理项
    - Mask受保护字符串
    - 分割破折号

### 替换空白字符
包括空格、换行、tab缩进等所有的空字符，在正则表达式中，我们可以使用`"\s+"`进行匹配。除此之外，在ASCII码中，第0～31号及第127号(共33个)是控制字符或通讯专用字符，如控制符：LF（换行）、CR（回车）、FF（换页）、DEL（删除）、BS（退格)、BEL（振铃）等；通讯专用字符：SOH（文头）、EOT（文尾）、ACK（确认）等，我们可以使用`"[\000-\037]"`进行匹配。

有了对应的正则表达式，在python中我们可以使用`re.sub`函数进行替换。


```python
import re

DEDUPLICATE_SPACE = r"\s+", r" "
ASCII_JUNK = r"[\000-\037]", r"" 

text = u" This is a test sentence  \t with useless blank chars\r.\x01"
print(text)

for regexp, substitution in [DEDUPLICATE_SPACE, ASCII_JUNK]:
    text = re.sub(regexp, substitution, text)

print(text)
```

    .
     This is a test sentence with useless blank chars .
    

### 去掉句子开头和结尾的空白字符
刚才将所有的空白字符替换为了空格，但是句子开头和结尾的空白字符也被替换成了空格，还没有被去掉，所以这里我们使用`strip()`方法去掉开头和结尾的空格。


```python
text = text.strip()
print(text)
```

    This is a test sentence with useless blank chars .
    

### 将常见标点、乱码等符号与词语分开
在Unicode字符集中，一部分字符会在我们的单词中出现，一部分则为标点符号以及其他的一些符号、乱码，如果我们的平行语料中这些字符通常与我们的单词连在一起，我们需要将它们与正常的单词分开。一个可行的方法是列举出所有可能出现在单词中字符（包括正常标点符号），除此之外的字符都在其两侧添加空格符号。幸运的是，moses中已经为我们搜集了这些字符，我们可以直接拿过来用（注意这里的标点符号不包含`.`，`.`，后续会单独处理）。


```python
with open("./assets/IsAlnum.txt") as f:
    IsAlnum = f.read()
    
PAD_NOT_ISALNUM = r"([^{}\s\.'\`\,\-])".format(IsAlnum), r" \1 "
regxp, substitution = PAD_NOT_ISALNUM

text = "This, is a sentence with weird\xbb symbols\u2026 appearing everywhere\xbf. Olso some normal punctuation such as?"

text = re.sub(regxp, substitution, text)
print(text.split())
```

    ['This,', 'is', 'a', 'sentence', 'with', 'weird', '»', 'symbols', '…', 'appearing', 'everywhere', '¿', '.', 'Olso', 'some', 'normal', 'punctuation', 'such', 'as', '?']
    

### 分割逗号`,`
这是刚才的遗留问题，由于`,`在用作子句划分的时候（I'm fine, thank you.）我们希望它与前后的单词分开，在某些情况下（如数字5,300）我们不希望将`,`与前后的字符分开。所以我们将`,`进行单独处理。


```python
with open("./assets/IsN.txt") as f:
    IsN = f.read()

COMMA_SEPARATE_1 = r"([^{}])[,]".format(IsN), r"\1 , "  # 若逗号前面不是数字，则分离逗号，如 hello,120 -> hello , 120
COMMA_SEPARATE_2 = r"[,]([^{}])".format(IsN), r" , \1"  # 若逗号后面不是数字，则分离逗号，如 120, hello -> 120 , hello
COMMA_SEPARATE_3 = r"([{}])[,]$".format(IsN), r"\1 , "  # 如果数字后匹配到结尾符，则分离逗号。 如120， -> 120 ,
COMMA_SEPARATE_4 = r"^[,]([{}])".format(IsN), r" \1, "  # 如果数字后匹配到结尾符，则分离逗号。 如120， -> 120 ,

text = ",This is a sentence 10, with number 5,300, and 5,"

# 此版本的实现可能会在此处创建额外的空格，但稍后会删除这些空格
for regxp, substitution in [COMMA_SEPARATE_1, COMMA_SEPARATE_2, COMMA_SEPARATE_3, COMMA_SEPARATE_4]:
    text = re.sub(regxp, substitution, text)
print(text.split())
```

    [',', 'This', 'is', 'a', 'sentence', '10', ',', 'with', 'number', '5,300', ',', 'and', '5', ',']
    

### 分割句号`.`
与`,`一样，`.`同样需要特殊的规则进行分割。需要考虑的情况有以下几种
1. 连续多个点号的情况（省略号）`.....`
2. 一个单独的大写字母跟一个`.` (通常出现在人名中，如`Aaron C. Courville`)
3. 其他的多字母人名，地名、机构名等缩写。 （如`Gov.`表示政府，Mr.代表某某先生）
4. 其他带`.`的缩写。（如`e.g.`表示举例，`i.e.`表示换句话说，`rev.`表示revision）
5. 一些`.`后面跟数字的情况（如`No`. `Nos.`），这种情况与前面的区别是只有当这些词后面跟随数字时才不是句子的结束，如`No.`也可能做否定的意思。
6. 月份的缩写。（如`Jan.` 表示一月，`Feb.`表示2月）

对于情况1，我们先匹配到多个`.`连续出现的情况，在其前后添加空格，并用Mask做标记（防止处理其他情况时对其造成不可预见的影响），待处理完其他情况后再将其还原。

对于后面几种情况，我们针对每一中情况建立一个前缀词表，如果`.`前面是这些词的话，就不讲`.`和前面的词分开。


```python
with open("./assets/nonbreaking_prefix.en") as f:
    NONBREAKING_PREFIXES  = []
    NUMERIC_ONLY_PREFIXES = []
    for line in f:
        line = line.strip()
        if line and not line.startswith("#"):
            if line.endswith("#NUMERIC_ONLY#"):
                NUMERIC_ONLY_PREFIXES.append(line.split()[0])
            if line not in NONBREAKING_PREFIXES:
                NONBREAKING_PREFIXES.append(line)

with open("./assets/IsAlpha.txt") as f:
    # IsAlnum = IsAlpha + IsN
    IsAlpha = f.read()
    
with open("./assets/IsLower.txt") as f:
    IsLower = f.read()

def isanyalpha(text):
    # 判断给定字符串中是否全是字母（非数字、符号）
    return any(set(text).intersection(set(IsAlpha)))

def islower(text):
    # 判断给定字符串中是否全部都是小写字母
    return not set(text).difference(set(IsLower))


def replace_multidots(text):
    # 处理情况1，对多个"."的情况作mask处理
    text = re.sub(r"\.([\.]+)", r" DOTMULTI\1", text)
    while re.search(r"DOTMULTI\.", text):
        text = re.sub(r"DOTMULTI\.([^\.])", r"DOTDOTMULTI \1", text)
        text = re.sub(r"DOTMULTI\.", "DOTDOTMULTI", text)
    return text

def handles_nonbreaking_prefixes(text):
    # 将文本拆分为标记以检查 "." 为结尾的部分是否符合拆分条件
    tokens = text.split()
    num_tokens = len(tokens)
    for i, token in enumerate(tokens):
        # 判断是否以"."结尾
        token_ends_with_period = re.search(r"^(\S+)\.$", token)
        if token_ends_with_period:  
            prefix = token_ends_with_period.group(1)

            # 处理情况2,3,4,6
            if (
                ("." in prefix and isanyalpha(prefix))
                or (
                    prefix in NONBREAKING_PREFIXES
                    and prefix not in NUMERIC_ONLY_PREFIXES
                )
                or (
                    i != num_tokens - 1
                    and tokens[i + 1]
                    and islower(tokens[i + 1][0])
                )
            ):
                pass  # 不做拆分处理

            # 处理情况 5
            elif (
                prefix in NUMERIC_ONLY_PREFIXES
                and (i + 1) < num_tokens
                and re.search(r"^[0-9]+", tokens[i + 1])
            ):
                pass  # 不做拆分处理
            else:  # 不在1-6中，做拆分处理
                tokens[i] = prefix + " ."
    return " ".join(tokens)  # Stitch the tokens back.

text = "This is a test sentence write on Sep. 6th. No. I am 123No. in all people. We are good at talk, swiming..."
text = replace_multidots(text)
text = handles_nonbreaking_prefixes(text)
print(text.split())
```

    ['This', 'is', 'a', 'test', 'sentence', 'write', 'on', 'Sep.', '6th', '.', 'No', '.', 'I', 'am', '123No.', 'in', 'all', 'people', '.', 'We', 'are', 'good', 'at', 'talk,', 'swiming', 'DOTDOTDOTMULTI']
    

### 处理`'`号缩写
在英文中，使用`'`号缩写非常常见，比如`I'm`,`You're`等等，他们其实是两个词，现在被缩写成了一个词，我们希望它被分割成`["I", "'m"]`，`["You", "'re"]`的形式。这个我们可以列几个简单的正则表达式进行处理。



```python
EN_SPECIFIC_1 = r"([^{alpha}])[']([^{alpha}])".format(alpha=IsAlpha), r"\1 ' \2"
EN_SPECIFIC_2 = (
    r"([^{alpha}{isn}])[']([{alpha}])".format(alpha=IsAlpha, isn=IsN),
    r"\1 ' \2",
)
EN_SPECIFIC_3 = r"([{alpha}])[']([^{alpha}])".format(alpha=IsAlpha), r"\1 ' \2"
EN_SPECIFIC_4 = r"([{alpha}])[']([{alpha}])".format(alpha=IsAlpha), r"\1 '\2"
EN_SPECIFIC_5 = r"([{isn}])[']([s])".format(isn=IsN), r"\1 '\2"

EN_SPECIFIC = [EN_SPECIFIC_1, EN_SPECIFIC_2, EN_SPECIFIC_3, EN_SPECIFIC_4, EN_SPECIFIC_5]

text = "I'm fine, thank you. And you?"
for regxp, substitution in EN_SPECIFIC:
    text = re.sub(regxp, substitution, text)

print(text.split())
```

    ['I', "'m", 'fine,', 'thank', 'you.', 'And', 'you?']
    

### 最后的收尾工作
刚才在分词的过程中，还留下了两个遗留问题，一个是我们对连续多个`.`进行了Mask，现在要对其进行还原。二是在上述过程中可能会在词与词之间产生多个空格，我们要把它们合并成一个。


```python
def restore_multidots(text):
    # 恢复对多个"."的mask
    while re.search(r"DOTDOTMULTI", text):
        text = re.sub(r"DOTDOTMULTI", r"DOTMULTI.", text)
    return re.sub(r"DOTMULTI", r".", text)
                  
DEDUPLICATE_SPACE = r"\s+", r" "
regxp, substitution = DEDUPLICATE_SPACE

text = "There are apple  , banana  DOTDOTDOTMULTI"
text = restore_multidots(text)
text = re.sub(regxp, substitution, text)
print(text)
```

    There are apple , banana ...
    

### 可选处理项
####  Mask 受保护的字符串
在分词过程中，有一些固定的字符串格式（比如url，日期，时间等），我们不希望把他们拆分开，而是希望将他们标注为统一标识符，以便于在翻译过程中减少词表的大小。


```python
# 匹配<\hello>标签
BASIC_PROTECTED_PATTERN_1 = r"<\/?\S+\/?>"

# 匹配xml的标签  <hello="2", hello2='3'>
BASIC_PROTECTED_PATTERN_2 = r'<\S+( [a-zA-Z0-9]+\="?[^"]")+ ?\/?>'
BASIC_PROTECTED_PATTERN_3 = r"<\S+( [a-zA-Z0-9]+\='?[^']')+ ?\/?>"
# 匹配邮箱
BASIC_PROTECTED_PATTERN_4 = r"[\w\-\_\.]+\@([\w\-\_]+\.)+[a-zA-Z]{2,}"
# 匹配url
BASIC_PROTECTED_PATTERN_5 = r"(http[s]?|ftp):\/\/[^:\/\s]+(\/\w+)*\/[\w\-\.]+"

BASIC_PROTECTED_PATTERNS = [
        BASIC_PROTECTED_PATTERN_1,
        BASIC_PROTECTED_PATTERN_2,
        BASIC_PROTECTED_PATTERN_3,
        BASIC_PROTECTED_PATTERN_4,
        BASIC_PROTECTED_PATTERN_5,
    ]

text = "this is a webpage https://stackoverflow.com/questions/6181381/how-to-print-variables-in-perl that kicks ass"

# Find the tokens that needs to be protected.
protected_tokens = [
    match.group()
    for protected_pattern in BASIC_PROTECTED_PATTERNS
    for match in re.finditer(protected_pattern, text, re.IGNORECASE)
]
# Apply the protected_patterns.
for i, token in enumerate(protected_tokens):
    substituition = "THISISPROTECTED" + str(i).zfill(3)
    text = text.replace(token, substituition)
    
print(text.split())
```

    ['this', 'is', 'a', 'webpage', 'THISISPROTECTED000', 'that', 'kicks', 'ass']
    

#### 分割破折号
对于型如`word-word`的连字符号，我们可以选择将它们分成两个词，中间的破折号用特殊符号标记（方便Detokenize）。思路还是使用上面的IsAlnum字符表，如果存在某个破折号，两边都是IsAlnum中的字符，则将破折号与两边的字符用空格隔开。


```python
AGGRESSIVE_HYPHEN_SPLIT = (
                  r"([{alphanum}])\-(?=[{alphanum}])".format(alphanum=IsAlnum),
                  r"\1 @-@ ",
              )

text = "This is a sentence with hyphen. pre-trained."
regxp, substitution = AGGRESSIVE_HYPHEN_SPLIT
text = re.sub(regxp, substitution, text)
print(text)
```

    This is a sentence with hyphen. pre @-@ trained.
    

## Putting them together


```python
import re
import os

def get_charset(charset_name):
    f = open(os.path.join("assets", charset_name + ".txt"))
    return f.read()


def get_nobreaking_prefix(lang="en"):
    f = open(os.path.join("assets", "nonbreaking_prefix." + lang))
    NONBREAKING_PREFIXES = []
    NUMERIC_ONLY_PREFIXES = []
    for line in f:
        line = line.strip()
        if line and not line.startswith("#"):
            if line.endswith("#NUMERIC_ONLY#"):
                NUMERIC_ONLY_PREFIXES.append(line.split()[0])
            if line not in NONBREAKING_PREFIXES:
                NONBREAKING_PREFIXES.append(line)
    f.close()
    return NONBREAKING_PREFIXES, NUMERIC_ONLY_PREFIXES


class MoseTokenizer(object):

    # 字符集
    IsAlnum = get_charset("IsAlnum")
    IsAlpha = get_charset("IsAlpha")
    IsLower = get_charset("IsLower")
    IsN = get_charset("IsN")

    # 步骤1 - 替换空白字符 相关正则表达式
    DEDUPLICATE_SPACE = r"\s+", r" "
    ASCII_JUNK = r"[\000-\037]", r""

    # 步骤2 - 将常见标点、乱码等符号与词语分开 相关正则表达式
    PAD_NOT_ISALNUM = r"([^{}\s\.'\`\,\-])".format(IsAlnum), r" \1 "

    # 步骤4 - 分割逗号 相关正则表达式
    # 若逗号前面不是数字，则分离逗号，如 hello,120 -> hello , 120
    COMMA_SEPARATE_1 = r"([^{}])[,]".format(IsN), r"\1 , "
    # 若逗号后面不是数字，则分离逗号，如 120, hello -> 120 , hello
    COMMA_SEPARATE_2 = r"[,]([^{}])".format(IsN), r" , \1"
    COMMA_SEPARATE_3 = r"([{}])[,]$".format(
        IsN), r"\1 , "  # 如果数字后匹配到结尾符，则分离逗号。 如120， -> 120 ,
    COMMA_SEPARATE_4 = r"^[,]([{}])".format(
        IsN), r" \1, "  # 如果数字后匹配到结尾符，则分离逗号。 如120， -> 120 ,

    COMMA_SEPARATE = [
        COMMA_SEPARATE_1,
        COMMA_SEPARATE_2,
        COMMA_SEPARATE_3,
        COMMA_SEPARATE_4
    ]

    # 步骤5 - 分割句号 受保护的前缀
    NONBREAKING_PREFIXES, NUMERIC_ONLY_PREFIXES = get_nobreaking_prefix(
        lang="en")

    # 步骤6 - 处理'号缩写 相关正则表达式
    EN_SPECIFIC_1 = r"([^{alpha}])[']([^{alpha}])".format(
        alpha=IsAlpha), r"\1 ' \2"
    EN_SPECIFIC_2 = (
        r"([^{alpha}{isn}])[']([{alpha}])".format(alpha=IsAlpha, isn=IsN),
        r"\1 ' \2",
    )
    EN_SPECIFIC_3 = r"([{alpha}])[']([^{alpha}])".format(
        alpha=IsAlpha), r"\1 ' \2"
    EN_SPECIFIC_4 = r"([{alpha}])[']([{alpha}])".format(
        alpha=IsAlpha), r"\1 '\2"
    EN_SPECIFIC_5 = r"([{isn}])[']([s])".format(isn=IsN), r"\1 '\2"

    EN_SPECIFIC = [
        EN_SPECIFIC_1,
        EN_SPECIFIC_1,
        EN_SPECIFIC_1,
        EN_SPECIFIC_1,
        EN_SPECIFIC_1
    ]

    # 可选步骤 Mask 受保护的字符串 相关的正则表达式
    BASIC_PROTECTED_PATTERN_1 = r"<\/?\S+\/?>"
    BASIC_PROTECTED_PATTERN_2 = r'<\S+( [a-zA-Z0-9]+\="?[^"]")+ ?\/?>'
    BASIC_PROTECTED_PATTERN_3 = r"<\S+( [a-zA-Z0-9]+\='?[^']')+ ?\/?>"
    BASIC_PROTECTED_PATTERN_4 = r"[\w\-\_\.]+\@([\w\-\_]+\.)+[a-zA-Z]{2,}"
    BASIC_PROTECTED_PATTERN_5 = r"(http[s]?|ftp):\/\/[^:\/\s]+(\/\w+)*\/[\w\-\.]+"

    BASIC_PROTECTED_PATTERNS = [
        BASIC_PROTECTED_PATTERN_1,
        BASIC_PROTECTED_PATTERN_2,
        BASIC_PROTECTED_PATTERN_3,
        BASIC_PROTECTED_PATTERN_4,
        BASIC_PROTECTED_PATTERN_5
    ]

    # 可选步骤 分割破折号 相关正则表达式
    AGGRESSIVE_HYPHEN_SPLIT = (
        r"([{alphanum}])\-(?=[{alphanum}])".format(alphanum=IsAlnum),
        r"\1 @-@ ",
    )

    def isanyalpha(self, text):
        # 判断给定字符串中是否全是字母（非数字、符号）
        return any(set(text).intersection(set(self.IsAlpha)))

    def islower(self, text):
        # 判断给定字符串中是否全部都是小写字母
        return not set(text).difference(set(self.IsLower))

    @staticmethod
    def replace_multidots(text):
        # 处理情况1，对多个"."的情况作mask处理
        text = re.sub(r"\.([\.]+)", r" DOTMULTI\1", text)
        while re.search(r"DOTMULTI\.", text):
            text = re.sub(r"DOTMULTI\.([^\.])", r"DOTDOTMULTI \1", text)
            text = re.sub(r"DOTMULTI\.", "DOTDOTMULTI", text)
        return text

    @staticmethod
    def restore_multidots(text):
        # 恢复对多个"."的mask
        while re.search(r"DOTDOTMULTI", text):
            text = re.sub(r"DOTDOTMULTI", r"DOTMULTI.", text)
        return re.sub(r"DOTMULTI", r".", text)

    def handles_nonbreaking_prefixes(self, text):
        # 将文本拆分为标记以检查 "." 为结尾的部分是否符合拆分条件
        tokens = text.split()
        num_tokens = len(tokens)
        for i, token in enumerate(tokens):
            # 判断是否以"."结尾
            token_ends_with_period = re.search(r"^(\S+)\.$", token)
            if token_ends_with_period:
                prefix = token_ends_with_period.group(1)

                # 处理情况2,3,4,6
                if (
                    ("." in prefix and self.isanyalpha(prefix))
                    or (
                        prefix in self.NONBREAKING_PREFIXES
                        and prefix not in self.NUMERIC_ONLY_PREFIXES
                    )
                    or (
                        i != num_tokens - 1
                        and tokens[i + 1]
                        and self.islower(tokens[i + 1][0])
                    )
                ):
                    pass  # 不做拆分处理

                # 处理情况 5
                elif (
                    prefix in self.NUMERIC_ONLY_PREFIXES
                    and (i + 1) < num_tokens
                    and re.search(r"^[0-9]+", tokens[i + 1])
                ):
                    pass  # 不做拆分处理
                else:  # 不在1-6中，做拆分处理
                    tokens[i] = prefix + " ."
        return " ".join(tokens)  # Stitch the tokens back.

    def tokenize(self,
                 text,
                 aggressive_dash_splits=False,  # 是否分割破折号 "-"
                 return_str=False,  # 返回字符串还是以list的形式返回
                 protected_patterns=None  # Mask 受保护的字符串 （以正则表达式list的形式传入）
                 ):

        # 步骤1 - 替换空白字符
        for regexp, substitution in [self.DEDUPLICATE_SPACE, self.ASCII_JUNK]:
            text = re.sub(regexp, substitution, text)

        # 可选步骤 Mask 受保护的字符串
        if protected_patterns:
            protecte_partterns.extend(self.BASIC_PROTECTED_PATTERNS)
        else:
            protecte_partterns = self.BASIC_PROTECTED_PATTERNS

        # Find the tokens that needs to be protected.
        protected_tokens = [
            match.group()
            for protected_pattern in self.BASIC_PROTECTED_PATTERNS
            for match in re.finditer(protected_pattern, text, re.IGNORECASE)
        ]
        # Apply the protected_patterns.
        for i, token in enumerate(protected_tokens):
            substituition = "THISISPROTECTED" + str(i).zfill(3)
            text = text.replace(token, substituition)

        # 步骤2 - 将常见标点、乱码等符号与词语分开 相关正则表达式
        regxp, substitution = self.PAD_NOT_ISALNUM
        text = re.sub(regxp, substitution, text)

        # 步骤3 - 去掉句子开头和结尾的空白字符
        text = text.strip()

        # 步骤4 - 分割逗号
        for regxp, substitution in self.COMMA_SEPARATE:
            text = re.sub(regxp, substitution, text)

        # 步骤5 - 分割句号
        text = self.replace_multidots(text)
        text = self.handles_nonbreaking_prefixes(text)

        # 步骤6 - 处理'号缩写
        for regxp, substitution in self.EN_SPECIFIC:
            text = re.sub(regxp, substitution, text)

        if aggressive_dash_splits:
            regxp, substitution = self.AGGRESSIVE_HYPHEN_SPLIT
            text = re.sub(regxp, substitution, text)

        # 收尾工作
        regxp, substitution = self.DEDUPLICATE_SPACE
        text = self.restore_multidots(text)
        text = re.sub(regxp, substitution, text)

        # 恢复受保护的字符串 Mask->原字符串.
        for i, token in enumerate(protected_tokens):
            substituition = "THISISPROTECTED" + str(i).zfill(3)
            text = text.replace(substituition, token)

        return text if return_str else text.split()
```


```python
test_sentences = [
    "this is a webpage https://stackoverflow.com/questions/6181381/how-to-print-variables-in-perl that kicks ass",
    "Sie sollten vor dem Upgrade eine Sicherung dieser Daten erstellen (wie unter Abschnitt 4.1.1, „Sichern aller Daten und Konfigurationsinformationen“ beschrieben).",
    "This ain't funny. It's actually hillarious, yet double Ls. | [] < > [ ] & You're gonna shake it off? Don't?",
    "This, is a sentence with weird\xbb symbols\u2026 appearing everywhere\xbf"
]

mt = MoseTokenizer()

for text in test_sentences:
    text = mt.tokenize(text)
    print(text)
```

    ['this', 'is', 'a', 'webpage', 'https://stackoverflow.com/questions/6181381/how-to-print-variables-in-perl', 'that', 'kicks', 'ass']
    ['Sie', 'sollten', 'vor', 'dem', 'Upgrade', 'eine', 'Sicherung', 'dieser', 'Daten', 'erstellen', '(', 'wie', 'unter', 'Abschnitt', '4.1.1', ',', '„', 'Sichern', 'aller', 'Daten', 'und', 'Konfigurationsinformationen', '“', 'beschrieben', ')', '.']
    ['This', "ain't", 'funny', '.', "It's", 'actually', 'hillarious', ',', 'yet', 'double', 'Ls', '.', '|', '[', ']', '<', '>', '[', ']', '&', "You're", 'gonna', 'shake', 'it', 'off', '?', "Don't", '?']
    ['This', ',', 'is', 'a', 'sentence', 'with', 'weird', '»', 'symbols', '…', 'appearing', 'everywhere', '¿']
    
