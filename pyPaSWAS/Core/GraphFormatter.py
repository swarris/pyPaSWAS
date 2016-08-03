''' Class file for adding alignments hits to a Neo4J graph database '''

from Formatters import DefaultFormatter
from neo4j.v1 import GraphDatabase, basic_auth

class GraphFormatter(DefaultFormatter):
    '''This Formatter is used to fill a Neo4J database
        from neo4j.v1 import GraphDatabase, basic_auth
        driver = GraphDatabase.driver("bolt://localhost", auth=basic_auth("neo4j", "Neo4J"))
        session = driver.session()

        session.run("CREATE (a:Person {name:'Arthur', title:'King'})")

        result = session.run("MATCH (a:Person) WHERE a.name = 'Arthur' RETURN a.name AS name, a.title AS title")
        for record in result:
            print("%s %s" % (record["title"], record["name"]))

        session.close()
    '''

    def __init__(self, logger, hitlist, outputfile, hostname, username, password):
        ''' The output file name will be used as prefix for contig names 
        '''
        DefaultFormatter.__init__(self, logger, hitlist, outputfile)
        self.outputfile = outputfile.split("/")[-1]
        driver = GraphDatabase.driver("bolt://{}".format(hostname), auth=basic_auth(username, password))
        self.session = driver.session()
        self.insert = []
        
    def _set_name(self):
        '''Name of the formatter. Used for logging'''
        self.name = 'Graph formatter'

    def _format_hit(self, hit):
        self.logger.debug('Formatting hit {0}'.format(hit.get_seq_id()))
        
        # check if node is in database
        node = self.session.run("match (r:Read {{name:'{}_{}'}}) return count(r) as c".format(self.outputfile, hit.get_seq_id()))
        for n in node:
            if n["c"] == 0:
                self.session.run("create (r:Read {{name:'{}_{}', length:{} }})".format(self.outputfile, hit.get_seq_id(), hit.sequence_info.original_length))
            
        if hit.get_target_id()[-2:] != 'RC':
            target_id = hit.get_target_id()
        else:
            target_id = hit.get_target_id()[:-3]
        
        node = self.session.run("match (r:Read {{name:'{}_{}'}}) return count(r) as c".format(self.outputfile, target_id))
        for n in node:
            if n["c"] == 0:
                self.session.run("create (r:Read {{name:'{}_{}', length:{} }})".format(self.outputfile, target_id, hit.target_info.original_length))
        
        self.insert.append(hit.get_graph_relation(self.outputfile, target_id))

    def print_results(self):
        '''sets, formats and prints the results to a file.'''
        self.logger.info('Adding results to graph database...')
        #format header and hit lines
        real_hits = self.hitlist.real_hits.values()
        for hit in real_hits:
            self._format_hit(hit)
        
        for i in self.insert:
            self.session.run(i)

        self.logger.debug('Results added to graph database')

