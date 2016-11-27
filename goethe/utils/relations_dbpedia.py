from SPARQLWrapper import SPARQLWrapper, JSON
import re
from .relation_builder import RelationBuilder

class DBpediaRelationBuilder(RelationBuilder):

    sparql = SPARQLWrapper("http://dbpedia.org/sparql")
    sparql.setReturnFormat(JSON)

    def query(self):
        raise NotImplementedError

    def sparql_query_results(self):
        self.sparql.setQuery(self.query())
        return self.sparql.query().convert()["results"]["bindings"]


class CountryCapital(DBpediaRelationBuilder):

    def query(self):
        return """
            SELECT distinct ?country_label, ?capital_label
            WHERE {
              ?country
                a dbo:Country ;
                a yago:WikicatMemberStatesOfTheUnitedNations .
              ?country dbo:capital ?capital .
              ?country rdfs:label ?country_label .
              ?capital rdfs:label ?capital_label .
              FILTER(LANG(?country_label) = "de") .
              FILTER(LANG(?capital_label) = "de")
            }
        """

    def relation_pairs(self):
        for result in self.sparql_query_results():
            country = result["country_label"]["value"]
            capital = result["capital_label"]["value"]
            capital = re.sub(r"\(.+\)", "", capital).strip()
            yield country, capital

class CountryCurrency(DBpediaRelationBuilder):

    def query(self):
        return """
            SELECT distinct ?country_label, ?currency_label
            WHERE {
              ?country
                a dbo:Country ;
                a yago:WikicatMemberStatesOfTheUnitedNations .
              ?country dbo:currency ?currency .
              ?country rdfs:label ?country_label .
              ?currency rdfs:label ?currency_label .
              FILTER(LANG(?country_label) = "de") .
              FILTER(LANG(?currency_label) = "de")
            }
        """

    def relation_pairs(self):
        for result in self.sparql_query_results():
            country = result["country_label"]["value"]
            capital = result["currency_label"]["value"]
            capital = re.sub(r"\(.+\)", "", capital).strip()
            yield country, capital



class CountryDemonym(DBpediaRelationBuilder):

    def query(self):
        return """
            SELECT distinct ?country_label, ?demonym_label
            WHERE {
              ?country
                a dbo:Country ;
                a yago:WikicatMemberStatesOfTheUnitedNations .
              ?country dbp:demonym ?capital .
              ?country rdfs:label ?country_label .
              ?capital rdfs:label ?demonym_label .
              FILTER(LANG(?country_label) = "de") .
              FILTER(LANG(?demonym_label) = "de" || LANG(?demonym_label) = "")
            }
        """

    def relation_pairs(self):
        for result in self.sparql_query_results():
            country = result["country_label"]["value"]
            demonym = result["demonym_label"]["value"]
            demonym = re.sub(r"\(.+\)", "", demonym).strip()
            yield country, demonym
