#build database class to give easy access to metainformation for a database system

#writing a class which we use to access the MySQL 

class Database:
 def _init_(self):
 "A class representation for MySQL database metadata"
 self.database = []
 
 #define fetchquery() and core methods
 # write internal function and pass all statements to it
 # Execute straightforward queries
def fetchquery(self, cursor, statement):
"Internal method that takes a statement and executes the query, returning the results"
try:
runit = cursor.execute(statement)
results = cursor.fetchall()
except MySQLdb.Error, e:
results = "The query you attempted failed. Please verify the information you have submitted and try again. The error message that was received reads: %s" %(e)
return results
def tables(self, cursor):
"Returns a list of the database tables"
statement = "SHOW TABLES"
header = ("Tables")
results = self.fetchquery(cursor, statement)
return header, results

# retrieve table status and structure
def tbstats(self):
"Returns the results of TABLE STATUS for the current db"
header = ("Name", "Engine", "Version", "Row_format", "Rows", "Avg_row_length", "Data_length", "Max_data_length", "Index_length", "Data_free", "Auto_increment", "Create_time", "Update_time", "Check_time", "Collation", "Checksum", "Create_options", "Comment")
statement = "SHOW TABLE STATUS"
results = self.fetchquery(statement)
return header, results

def describe(self, tablename):
"Returns the column structure of a specified table"
header = ("Field", "Type", "Null", " Key", "Default", "Extra")
statement = "SHOW COLUMNS FROM %s" %(tablename)
results = self.fetchquery(statement)
return header, results

#retrieve table status and structure
def tbstats(self):
"Returns the results of TABLE STATUS for the current db"
header = ("Name", "Engine", "Version", "Row_format", "Rows", "Avg_row_length", "Data_length", "Max_data_length", "Index_length", "Data_free", "Auto_increment", "Create_time", "Update_time", "Check_time", "Collation", "Checksum", "Create_options", "Comment")
statement = "SHOW TABLE STATUS"
results = self.fetchquery(statement)
return header, results
def describe(self, tablename):
"Returns the column structure of a specified table"
header = ("Field", "Type", "Null", " Key", "Default", "Extra")
statement = "SHOW COLUMNS FROM %s" %(tablename)
results = self.fetchquery(statement)
return header, results
