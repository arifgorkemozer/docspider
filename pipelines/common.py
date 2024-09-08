# generic definitions for writing csv/tsv files
TAB_CHAR = '\t'
NEWLINE_CHAR = '\n'
SEMICOLON_CHAR = ';'
COLON_CHAR = ':'
DOUBLE_QUOTE_CHAR = '"'
SINGLE_QUOTE_CHAR = '\''
SPACE_CHAR = ' '
EMPTY_STR = ""

EXPERIMENT_NL2SQL = "nl2sql"
EXPERIMENT_NL2MONGO = "nl2mongo"

SENSITIVITY_SAME = "same" # same number of rows, same row order, same columns
SENSITIVITY_EXTRA_FIELDS = "extra_fields" # same number of rows, same row order, more columns than expected (projection error)
SENSITIVITY_UNORDERED = "unordered" # same number of rows, different row order, same columns (ordering error)
SENSITIVITY_UNORDERED_EXTRA_FIELDS = "unordered_extra_fields" # same number of rows, different row order, more columns than expected (ordering and projection error)
SENSITIVITY_OPTIONS = [SENSITIVITY_SAME, SENSITIVITY_EXTRA_FIELDS, SENSITIVITY_UNORDERED, SENSITIVITY_UNORDERED_EXTRA_FIELDS]

def remove_double_space(word):
    current = None
    next = word

    while current != next:
        current = next
        next = next.replace("  ", " ")
    
    return current