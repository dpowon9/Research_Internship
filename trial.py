S = ["mobile", "mouse", "moneypot", "mousepad", "monitor"]


def suggest(In, g):
    In = In.lower()
    if len(In) < 2:
        return []
    g = [word.lower() for word in g]
    g.sort()
    box = []
    for word in g:
        if word.startswith(In):
            box.append(word)
    return box[0:3]


print(suggest('mob', S))
S2 = "jack and jill went to the market to buy bread and cheese cheese is jack favorite food food market jill Bread Buy Buy henry henry Jack jack"
exc = ["and", "he", "the", "to", "is"]


def common(string, exclude):
    string = [word for word in string.split() if word not in exclude]
    new_dict = {}
    for i in string:
        if i not in new_dict:
            new_dict[i] = string.count(i)
    occur = max(list(new_dict.values()))
    res = [i for i in new_dict.keys() if new_dict[i] == occur]
    return res


print(common(S2, exc))
