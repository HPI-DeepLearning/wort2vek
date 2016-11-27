class RelationBuilder:
    """
    Creates relation files with a list of tuples with the same relation. These
    files can be used to create quadtruples for accuracy testing of word vectors.
    """

    def relation_pairs(self):
        raise NotImplementedError

    def is_single_word(self, word):
        return " " not in word.strip()

    def are_single_words(self, w1, w2):
        return self.is_single_word(w1) and self.is_single_word(w2)

    def relation_lines(self):
        for (w1, w2) in self.relation_pairs():
            if self.are_single_words(w1, w2):
                yield w1 + "," + w2

    def write_to_file(self, filename):
        with open(filename, 'w') as f:
            f.writelines(l + '\n' for l in self.relation_lines())
