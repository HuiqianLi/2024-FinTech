# 选手需要设计代码加载知识库中的内容，并完成相应的逻辑问题回答。
# Capacity(A) && ResidenceSP(A) && InvestThreshold(A) => SouthPass(A)
 #encoding=utf-8
# 注意:    rules.txt, facts.txt, input.txt 会出现多余的空行或空格
#         input.txt每行对应一个 query, 可能会出现多余空格; output.txt不能出现多余空格或空行，每个 query 对应一行

def reasLine(line):
    # 去掉所有空格
    line = line.replace(' ', '')
    # 将line按' && '或者' => '分割成列表
    query = line.strip().split('&&')
    if len(query) == 1:
        query = line.strip().split('=>')
    ans = query[-1].split('=>')[-1]
    query[-1] = query[-1].split('=>')[0]
    return query, ans

def readRules():
    rules = {}
    with open("rules.txt", "r") as ruleFile:
        for line in ruleFile:
            # 忽略空行
            if line.strip():
                query, ans = reasLine(line)
                # 将解析后的规则添加到字典中
                if ans not in rules:
                    rules[ans] = []
                rules[ans].append(query)
    return rules


def readFacts():
    facts_list = []  # 初始化一个空列表用于存储读取的行
    with open("facts.txt", "r") as factFile:
        # 逐行读取文件
        for line in factFile:
            # 跳过空行
            if line.strip():  # line.strip() 移除行首尾的空白字符，如果行不为空则继续
                facts_list.append(line.strip())  # 添加非空行到列表中
    return facts_list

def solve(line):
    res = []
    query, ans = reasLine(line)
    for q in query:
        if q in facts:
            res.append("True")
            memo[q] = True
        else:
            res.append(recursive_solve(query, q, facts, rules))
    res.append(recursive_solve(query, ans, facts, rules))
    return ','.join(res)

def recursive_solve(query, ans, facts, rules):
    ans_ = ans.split('(')[0]
    ans_variable = ans.split('(')[1][:-1]
    two_variable = False
    # 有两个变量的情况
    if ',' in ans_variable:
        ans_variable = ans_variable.split(',')
        two_variable = True
    
    if not two_variable:
        for rule, conditions in rules.items():
            rule_ = rule.split('(')[0]
            if rule_ == ans_:
                for c in conditions:
                    # c = ['Product(x)', 'ApprovedHKMC(x)']
                    if all(deduce(query, ci, facts, rules, ans_variable, memo) for ci in c):
                        return "True"
        return "False"
    else:
        for rule, conditions in rules.items():
            rule_ = rule.split('(')[0]
            rule_variable = rule.split('(')[1][:-1]
            rule_variable = rule_variable.split(',')
            if rule_ == ans_:
                for c in conditions:
                    # c = ['SouthPass(A)', 'ProductSP(x)']
                    flag = True
                    for ci in c:
                        ci_variable = ci.split('(')[1][:-1]
                        ans_variable_i = ans_variable[rule_variable.index(ci_variable)]
                        if not deduce(query, ci, facts, rules, ans_variable_i, memo):
                            flag = False
                            # return "False"
                    if flag:
                        return "True"
        return "False"

def deduce(query, fact, facts, rules, variables, memo):
    fact_variable = fact.split('(')[1][:-1]
    fact = fact.replace(fact_variable, variables)
    # 如果fact已经在记忆化字典中，则直接返回结果
    if fact in memo.keys():
        return memo[fact]
    # 检查fact是否可以从facts直接得到
    # if fact in query and fact in facts:
    if fact in facts:
        memo[fact] = True
        return True
    # 尝试使用知识库中的规则来推理fact
    for rule, conditions in rules.items():
        rule_variable = rule.split('(')[1][:-1]
        rule = rule.replace(rule_variable, variables)
        if rule == fact:
            for c in conditions:
                # 检查所有条件是否都为真
                if all(deduce(query, ci, facts, rules, variables, memo) for ci in c):
                    # 如果所有条件都为真，则fact可以被推导出来
                    memo[fact] = True
                    return True
    # 如果没有规则可以推导出fact，则返回False，并存储结果
    memo[fact] = False
    return False
    
if __name__ == '__main__':
    # 调用readRules, readFacts 加载知识库
    facts = readFacts()

    rules = readRules()

    # 回答每个查询
    # 记忆化字典，用于存储已经推导出的事实
    memo = {}
    with open("input.txt", "r") as inFile, open("output.txt", "w") as outFile:
        for line in inFile:
            # 遇到空行直接填换行符
            if not line.strip():
                continue
            # print(line.strip())
            outFile.write(solve(line) + "\n")
            # break