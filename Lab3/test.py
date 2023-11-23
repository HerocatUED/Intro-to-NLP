import numpy as np

def get_oracle(stack, buf, ex):
    # TODO: 根据当前状态，返回应该执行的操作编号（对应__init__中的trans），若无操作则返回None。
    if len(stack) < 2:
        return None if len(buf) == 0 else 'S'
    s1, s2 = stack[-1], stack[-2]
    h1, h2 = ex['head'][s1], ex['head'][s2]
    if s2 > 0 and h2 == s1:
        return 'L'
    elif s2 >= 0 and h1 == s2 and (not any([x for x in buf if ex['head'][x] == s1])):
        return 'R'
    else:
        return None if len(buf) == 0 else 'S'
    

stack = ['root']
buf = []
ex = {'word':[0], 'head':[-1], 'label':[0]}
print(get_oracle(stack, buf, ex))