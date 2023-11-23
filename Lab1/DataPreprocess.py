import pandas as pd
import string
import re


def preprocess(filename: str):
    file = pd.read_csv(filename)
    file = file.drop_duplicates()
    file = file.dropna()
    file['data'] = file['data'].apply(trans)
    return file


def trans(text: str):
    """
    1. 除去标点、数字、换行符
    2. 除去多余的空格
    3. 转化为小写形式
    """
    del_str = string.punctuation+string.digits+"\n"
    replace = " "*len(del_str)
    tran_tab = str.maketrans(del_str, replace)
    text = text.translate(tran_tab)
    new_text = re.sub(r"\s+", " ", text)
    new_text = new_text.strip()
    new_text = new_text.lower()
    return new_text
