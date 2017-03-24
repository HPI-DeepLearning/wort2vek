class Window:

    PARSE_TREE = False

    def __init__(self, **kwargs):
        pass

    def window_sort(self, tokens, n):
        i = n - 1
        j = n + 1
        l = []
        while i >= 0 or j < len(tokens):
            if i >= 0:
                l.append(tokens[i])
            if j < len(tokens):
                l.append(tokens[j])
            i -= 1
            j += 1
        return l

    def lines(self, doc):
        tokens = [t.text for t in doc]
        for i in range(len(tokens)):
            yield [tokens[i]]s + self.window_sort(tokens, i)


if __name__ == '__main__':
    t = ['Ich', 'bin', 'der', 'Nico', '.']
    w = Window()
    ls = w.token_list(t)
    for l in ls:
        print(l)
