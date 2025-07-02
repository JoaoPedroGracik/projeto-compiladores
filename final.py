import re
import tkinter as tk
from tkinter import scrolledtext
from graphviz import Digraph
from PIL import Image, ImageTk

# Tokens e expressoes regulares
TOKEN_REGEX = [
    (r'int\b', 'INT'),
    (r'if\b', 'IF'),
    (r'else\b', 'ELSE'),
    (r'while\b', 'WHILE'),
    (r'for\b', 'FOR'),
    (r'return\b', 'RETURN'),
    (r'true\b|false\b', 'BOOLEAN'),
    (r'[a-zA-Z_][a-zA-Z0-9_]*', 'IDENTIFIER'),
    (r'\d+', 'INTEGER'),
    (r'==', 'EQUALS'),
    (r'=', 'ASSIGN'),
    (r'!=', 'NOT_EQUALS'),
    (r'!', 'NOT'),
    (r'&&', 'AND'),
    (r'\|\|', 'OR'),
    (r'<=', 'LESS_EQUAL'),
    (r'>=', 'GREATER_EQUAL'),
    (r'<', 'LESS_THAN'),
    (r'>', 'GREATER_THAN'),
    (r'\+', 'PLUS'),
    (r'-', 'MINUS'),
    (r'\*', 'MULTIPLY'),
    (r'/', 'DIVIDE'),
    (r';', 'SEMICOLON'),
    (r'\(', 'LPAREN'),
    (r'\)', 'RPAREN'),
    (r'\[', 'LBRACKET'),
    (r'\]', 'RBRACKET'),
    (r'\{', 'LBRACE'),
    (r'\}', 'RBRACE'),
    (r',', 'COMMA'),
    (r'\?', 'QUESTION'),
    (r':', 'COLON'),
    (r'"[^"]*"', 'STRING'),
    (r'\s+', None),
]

# Analisador Lexico
def lexer(code):
    tokens = []
    position = 0
    while position < len(code):
        match = None
        for token_regex, token_type in TOKEN_REGEX:
            regex = re.compile(token_regex)
            match = regex.match(code, position)
            if match:
                if token_type:
                    tokens.append((token_type, match.group(0)))
                position = match.end(0)
                break
        if not match:
            raise ValueError(f"Erro léxico: Caractere inválido '{code[position]}' encontrado em Linha 1, coluna {position + 1}.")
    return tokens

# Analisador Sintatico
class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0

    def current_token(self):
        return self.tokens[self.pos] if self.pos < len(self.tokens) else ('EOF', 'EOF')

    def eat(self, token_type):
        if self.current_token()[0] == token_type:
            self.pos += 1
        else:
            raise SyntaxError(f"Erro sintático: Esperado {token_type} mas encontrado {self.current_token()[0]}")


    def parse_funcao(self):
        if self.current_token()[0] == 'INT':
            tipo = 'int'
            self.eat('INT')
        elif self.current_token()[0] == 'IDENTIFIER' and self.current_token()[1] in ('string', 'bool'):
            tipo = self.current_token()[1]
            self.eat('IDENTIFIER')
        else:
            raise SyntaxError(f"Erro sintático: tipo de função '{self.current_token()[1]}' inválido.")

        nome = self.current_token()[1]
        self.eat('IDENTIFIER')

        self.eat('LPAREN')
        params = self.parse_parametros()
        self.eat('RPAREN')

        bloco = self.parse_bloco()

        return ('FUNCAO', tipo, nome, params, bloco)

    def parse_parametros(self):
        params = []
        if self.current_token()[0] in ('INT', 'IDENTIFIER'):

            while True:
                if self.current_token()[0] == 'INT':
                    tipo = 'int'
                    self.eat('INT')
                elif self.current_token()[0] == 'IDENTIFIER' and self.current_token()[1] in ('string', 'bool'):
                    tipo = self.current_token()[1]
                    self.eat('IDENTIFIER')
                else:
                    raise SyntaxError("Parâmetro com tipo inválido.")

                nome = self.current_token()[1]
                self.eat('IDENTIFIER')

                params.append((tipo, nome))

                if self.current_token()[0] != 'COMMA':
                    break
                self.eat('COMMA')

        return params


    def parse_programa(self):
        elementos = []
        while self.pos < len(self.tokens):
            # Checa se é uma função ou uma variável
            if self.current_token()[0] == 'INT' or (
                self.current_token()[0] == 'IDENTIFIER' and self.current_token()[1] in ('string', 'bool')
            ):
                tipo_token = self.current_token()
                next_token = self.tokens[self.pos + 2] if self.pos + 2 < len(self.tokens) else ('EOF', '')

                # INT IDENTIFIER LPAREN → função
                if next_token[0] == 'LPAREN':
                    elementos.append(self.parse_funcao())
                else:
                    elementos.append(self.parse_comando())
            else:
                elementos.append(self.parse_comando())
        return ('PROGRAMA', *elementos)

    def parse_lista_de_comandos(self):
        comandos = []
        while self.pos < len(self.tokens) and self.current_token()[0] != 'RBRACE':
            comandos.append(self.parse_comando())
        return comandos

    def parse_comando(self):
        token = self.current_token()
        if token[0] == 'INT':
            return self.parse_declaracao()
        elif token[0] == 'IDENTIFIER' and token[1] in ('string', 'bool'):
            return self.parse_declaracao()
        elif token[0] == 'IDENTIFIER':
            # Pode ser atribuição normal, vetor, matriz ou chamada
            return self.parse_atribuicao_ou_chamada()
        elif token[0] == 'IF':
            return self.parse_if_comando()
        elif token[0] == 'WHILE':
            return self.parse_while_comando()
        elif token[0] == 'FOR':
            return self.parse_for_comando()
        elif token[0] == 'RETURN':
            return self.parse_return_comando()
        elif token[0] == 'LBRACE':
            return self.parse_bloco()
        else:
            return self.parse_expressao_comando()

    def parse_primaria(self):
        token = self.current_token()
        if token[0] == 'IDENTIFIER':
            nome = token[1]
            self.eat('IDENTIFIER')

            if self.current_token()[0] == 'LBRACKET':
                self.eat('LBRACKET')
                index1 = self.parse_expressao()
                self.eat('RBRACKET')

                if self.current_token()[0] == 'LBRACKET':
                    self.eat('LBRACKET')
                    index2 = self.parse_expressao()
                    self.eat('RBRACKET')

                    if self.current_token()[0] == 'ASSIGN':
                        self.eat('ASSIGN')
                        value = self.parse_expressao()
                        return ('MATRIX_ASSIGN', nome, index1, index2, value)
                    return ('MATRIX_ACCESS', nome, index1, index2)

                else:
                    if self.current_token()[0] == 'ASSIGN':
                        self.eat('ASSIGN')
                        value = self.parse_expressao()
                        return ('ARRAY_ASSIGN', nome, index1, value)
                    return ('ARRAY_ACCESS', nome, index1)

            else:
                if self.current_token()[0] == 'ASSIGN':
                    self.eat('ASSIGN')
                    value = self.parse_expressao()
                    return ('ASSIGN', ('IDENTIFIER', nome), value)
                return ('IDENTIFIER', nome)

        # (outros casos: INTEGER, STRING, BOOLEAN, etc.)

    def parse_declaracao(self):
        # tipo
        if self.current_token()[0] == 'INT':
            tipo = 'int'
            self.eat('INT')
        elif self.current_token()[0] == 'IDENTIFIER' and self.current_token()[1] in ('string', 'bool'):
            tipo = self.current_token()[1]
            self.eat('IDENTIFIER')
        else:
            raise SyntaxError(f"Erro sintático: tipo '{self.current_token()[1]}' inválido.")

        # nome
        nome = self.current_token()[1]
        self.eat('IDENTIFIER')

        # vetor / matriz
        if self.current_token()[0] == 'LBRACKET':
            self.eat('LBRACKET')
            tamanho1 = self.current_token()[1]
            self.eat('INTEGER')
            self.eat('RBRACKET')

            if self.current_token()[0] == 'LBRACKET':
                self.eat('LBRACKET')
                tamanho2 = self.current_token()[1]
                self.eat('INTEGER')
                self.eat('RBRACKET')
                self.eat('SEMICOLON')
                return ('DECL_MATRIX', tipo, nome, tamanho1, tamanho2)
            else:
                self.eat('SEMICOLON')
                return ('DECL_VECTOR', tipo, nome, tamanho1)

        # atribuição
        elif self.current_token()[0] == 'ASSIGN':
            self.eat('ASSIGN')
            expr = self.parse_expressao()
            self.eat('SEMICOLON')
            return ('DECL_ASSIGN', tipo, nome, expr)

        # simples
        else:
            self.eat('SEMICOLON')
            return ('DECL', tipo, nome)

    
    
    def parse_if_comando(self):
        self.eat('IF')
        self.eat('LPAREN')
        cond = self.parse_expressao()
        self.eat('RPAREN')
        true_block = self.parse_comando()

        if self.current_token()[0] == 'ELSE':
            self.eat('ELSE')
            false_block = self.parse_comando()
            return ('IF', cond, true_block, false_block)
        else:
            return ('IF', cond, true_block)

    def parse_while_comando(self):
        self.eat('WHILE')
        self.eat('LPAREN')
        expr = self.parse_expressao()
        self.eat('RPAREN')
        comando = self.parse_comando()
        return ('WHILE', expr, comando)
    
    def parse_for_comando(self):
        self.eat('FOR')
        self.eat('LPAREN')

        # Parte 1: inicialização (declaração ou atribuição)
        if self.current_token()[0] == 'INT':
            init = self.parse_declaracao()
        elif self.current_token()[0] == 'IDENTIFIER':
            init = self.parse_atribuicao_ou_chamada()
        else:
            raise SyntaxError("Esperado declaração ou atribuição na inicialização do for")

        # Parte 2: condição
        cond = self.parse_expressao()
        self.eat('SEMICOLON')

        # Parte 3: incremento (atribuição ou expressão)
        if self.current_token()[0] == 'IDENTIFIER':
            increment = self.parse_atribuicao_ou_chamada()
        else:
            increment = self.parse_expressao()

        self.eat('RPAREN')  # Correto: RPAREN vem após incremento

        # Bloco do for
        bloco = self.parse_comando()

        return ('FOR', init, cond, increment, bloco)

    def parse_return_comando(self):
        self.eat('RETURN')
        expr = self.parse_expressao()
        self.eat('SEMICOLON')
        return ('RETURN', expr)

    def parse_expressao_comando(self):
        node = self.parse_expressao()
        self.eat('SEMICOLON')
        return node

    def parse_bloco(self):
        self.eat('LBRACE')
        comandos = []
        while self.current_token()[0] != 'RBRACE':
            comandos.append(self.parse_comando())
        self.eat('RBRACE')
        return ('BLOCK', comandos)

    def parse_expressao(self):
        node = self.parse_or()

        if self.current_token()[0] == 'QUESTION':
            self.eat('QUESTION')
            true_expr = self.parse_expressao()
            self.eat('COLON')
            false_expr = self.parse_expressao()
            node = ('TERNARY', node, true_expr, false_expr)

        return node

    def parse_or(self):
        node = self.parse_and()
        while self.current_token()[0] == 'OR':
            op = self.current_token()[0]
            self.eat(op)
            right = self.parse_and()
            node = ('BIN_OP', op, node, right)
        return node

    def parse_and(self):
        node = self.parse_relacional()
        while self.current_token()[0] == 'AND':
            op = self.current_token()[0]
            self.eat(op)
            right = self.parse_relacional()
            node = ('BIN_OP', op, node, right)
        return node

    def parse_relacional(self):
        left = self.parse_aditiva()
        token = self.current_token()
        if token[0] in ('LESS_THAN', 'GREATER_THAN', 'LESS_EQUAL', 'GREATER_EQUAL', 'EQUALS', 'NOT_EQUALS'):
            op = token[0]
            self.eat(op)
            right = self.parse_aditiva()
            return ('BIN_OP', op, left, right)
        return left

    def parse_aditiva(self):
        node = self.parse_multiplicativa()
        while self.current_token()[0] in ('PLUS', 'MINUS'):
            op = self.current_token()[0]
            self.eat(op)
            right = self.parse_multiplicativa()
            node = ('BIN_OP', op, node, right)
        return node

    def parse_multiplicativa(self):
        node = self.parse_primaria()
        while self.current_token()[0] in ('MULTIPLY', 'DIVIDE'):
            op = self.current_token()[0]
            self.eat(op)
            right = self.parse_primaria()
            node = ('BIN_OP', op, node, right)
        return node

    def parse_primaria(self):
        token = self.current_token()
        if token[0] == 'IDENTIFIER':
            next_token = self.tokens[self.pos + 1] if self.pos + 1 < len(self.tokens) else ('EOF', '')
            if next_token[0] == 'LPAREN':
                return self.parse_chamada_funcao()
            else:
                self.eat('IDENTIFIER')
                return ('IDENTIFIER', token[1])
        elif token[0] == 'INTEGER':
            self.eat('INTEGER')
            return ('INTEGER', token[1])
        elif token[0] == 'LPAREN':
            self.eat('LPAREN')
            expr = self.parse_expressao()
            self.eat('RPAREN')
            return expr
        elif token[0] == 'STRING':
            self.eat('STRING')
            return ('STRING', token[1])
        elif token[0] == 'BOOLEAN':
            self.eat('BOOLEAN')
            return ('BOOLEAN', token[1])
        elif token[0] == 'NOT':
            self.eat('NOT')
            expr = self.parse_primaria()
            return ('UNARY_OP', 'NOT', expr)
        else:
            raise SyntaxError(f"Erro sintático: token inesperado {token[0]}")
        
    def parse_chamada_funcao(self):
        func_name = self.current_token()[1]
        self.eat('IDENTIFIER')
        self.eat('LPAREN')
        args = []
        if self.current_token()[0] != 'RPAREN':
            while True:
                arg = self.parse_expressao()
                args.append(arg)
                if self.current_token()[0] == 'COMMA':
                    self.eat('COMMA')
                else:
                    break
        self.eat('RPAREN')
        return ('CALL', func_name, args)
    
    def parse_atribuicao_ou_chamada(self):
        nome = self.current_token()[1]
        self.eat('IDENTIFIER')

        if self.current_token()[0] == 'LBRACKET':
            self.eat('LBRACKET')
            index1 = self.parse_expressao()
            self.eat('RBRACKET')

            if self.current_token()[0] == 'LBRACKET':
                self.eat('LBRACKET')
                index2 = self.parse_expressao()
                self.eat('RBRACKET')
                self.eat('ASSIGN')
                value = self.parse_expressao()
                self.eat('SEMICOLON')
                return ('MATRIX_ASSIGN', nome, index1, index2, value)
            else:
                self.eat('ASSIGN')
                value = self.parse_expressao()
                self.eat('SEMICOLON')
                return ('ARRAY_ASSIGN', nome, index1, value)

        elif self.current_token()[0] == 'ASSIGN':
            self.eat('ASSIGN')
            value = self.parse_expressao()
            self.eat('SEMICOLON')
            return ('ASSIGN', ('IDENTIFIER', nome), value)

        elif self.current_token()[0] == 'LPAREN':
            # chamada de função
            self.eat('LPAREN')
            args = []
            if self.current_token()[0] != 'RPAREN':
                while True:
                    arg = self.parse_expressao()
                    args.append(arg)
                    if self.current_token()[0] == 'COMMA':
                        self.eat('COMMA')
                    else:
                        break
            self.eat('RPAREN')
            self.eat('SEMICOLON')
            return ('CALL', nome, args)

        else:
            raise SyntaxError(f"Erro sintático: Esperado atribuição, chamada ou acesso, mas encontrado {self.current_token()[0]}")


# Impressao textual
def print_ast(node, level=0):
    indent = "  " * level
    if isinstance(node, tuple):
        result = f"{indent}{node[0]}\n"
        for child in node[1:]:
            result += print_ast(child, level + 1)
        return result
    else:
        return f"{indent}{node}\n"

# Geracao grafica da arvore
def build_graphviz_ast(node, graph=None, parent=None, counter=[0]):
    if graph is None:
        graph = Digraph()
        graph.attr(rankdir='TB')

    node_id = f"node{counter[0]}"
    counter[0] += 1
    label = str(node[0]) if isinstance(node, tuple) else str(node)
    graph.node(node_id, label)

    if parent:
        graph.edge(parent, node_id)

    if isinstance(node, tuple):
        for child in node[1:]:
            build_graphviz_ast(child, graph, node_id, counter)

    return graph

tac = []
temp_counter = 0
label_counter = 0

def new_temp():
    global temp_counter
    temp_counter += 1
    return f"t{temp_counter}"

def new_label():
    global label_counter
    label_counter += 1
    return f"L{label_counter}"

def generate_TAC(node):
    if isinstance(node, tuple):
        kind = node[0]

        if kind == 'BIN_OP':
            left = generate_TAC(node[2])
            right = generate_TAC(node[3])
            temp = new_temp()
            tac.append(f"{temp} = {left} {node[1]} {right}")
            return temp

        elif kind == 'ASSIGN':
            var = generate_TAC(node[1])
            value = generate_TAC(node[2])
            tac.append(f"{var} = {value}")
            return var

        elif kind == 'DECL_ASSIGN':
            var = node[2]
            value = generate_TAC(node[3])
            temp = new_temp()
            tac.append(f"{temp} = {value}")
            tac.append(f"{var} = {temp}")
            return var

        elif kind == 'INTEGER':
            return node[1]
        elif kind == 'IDENTIFIER':
            return node[1]
        elif kind == 'STRING':
            return node[1]
        elif kind == 'BOOLEAN':
            return node[1]
        
        elif kind == 'BLOCK' or kind == 'PROGRAMA':
            for subnode in node[1:]:
                generate_TAC(subnode)
                
        elif kind == 'CALL':
            args_temps = [generate_TAC(arg) for arg in node[2]]
            temp = new_temp()
            tac.append(f"{temp} = call {node[1]}({', '.join(args_temps)})")
            return temp
        
        elif node[0] == 'TERNARY':
            cond = generate_TAC(node[1])
            true_label = new_label()
            false_label = new_label()
            end_label = new_label()
            temp = new_temp()

            tac.append(f"if {cond} goto {true_label}")
            tac.append(f"goto {false_label}")

            tac.append(f"{true_label}:")
            t_true = generate_TAC(node[2])
            tac.append(f"{temp} = {t_true}")
            tac.append(f"goto {end_label}")

            tac.append(f"{false_label}:")
            t_false = generate_TAC(node[3])
            tac.append(f"{temp} = {t_false}")

            tac.append(f"{end_label}:")

            return temp
        
        elif node[0] == 'UNARY_OP' and node[1] == 'NOT':
            val = generate_TAC(node[2])
            temp = new_temp()
            tac.append(f"{temp} = !{val}")
            return temp

        elif node[0] == 'ARRAY_ASSIGN':
            nome = node[1]
            index = generate_TAC(node[2])
            value = generate_TAC(node[3])
            tac.append(f"{nome}[{index}] = {value}")
            return None

        elif node[0] == 'MATRIX_ASSIGN':
            nome = node[1]
            row = generate_TAC(node[2])
            col = generate_TAC(node[3])
            offset = new_temp()
            tac.append(f"{offset} = {row} * N_COLS + {col}")  # N_COLS: substitua pelo valor real, ex: 3
            value = generate_TAC(node[4])
            tac.append(f"{nome}[{offset}] = {value}")
            return None

        elif node[0] == 'ARRAY_ACCESS':
            nome = node[1]
            index = generate_TAC(node[2])
            temp = new_temp()
            tac.append(f"{temp} = {nome}[{index}]")
            return temp

        elif node[0] == 'MATRIX_ACCESS':
            nome = node[1]
            row = generate_TAC(node[2])
            col = generate_TAC(node[3])
            offset = new_temp()
            tac.append(f"{offset} = {row} * N_COLS + {col}")
            temp = new_temp()
            tac.append(f"{temp} = {nome}[{offset}]")
            return temp
        
        elif node[0] == 'DECL_VECTOR':
            nome = node[2]
            tamanho = node[3]
            tac.append(f"ALLOC {nome}[{tamanho}]")
            return None

        elif node[0] == 'DECL_MATRIX':
            nome = node[2]
            linhas = node[3]
            colunas = node[4]
            tac.append(f"ALLOC {nome}[{linhas}][{colunas}]")
            return None

        elif kind == 'RETURN':
            value = generate_TAC(node[1])
            tac.append(f"return {value}")
            return value


# ---------------------------- Analisador Semântico ----------------------------
class SymbolTable:
    def __init__(self, parent=None):
        self.symbols = {}
        self.parent = parent

    def define(self, name, symbol_type):
        if name in self.symbols:
            raise ValueError(f"Erro semântico: '{name}' já declarado neste escopo.")
        self.symbols[name] = symbol_type

    def lookup(self, name):
        if name in self.symbols:
            return self.symbols[name]
        elif self.parent:
            return self.parent.lookup(name)
        else:
            return None
        
class FunctionTable:
    def __init__(self):
        self.functions = {}

    def define(self, name, return_type, param_types):
        self.functions[name] = (return_type, param_types)

    def lookup(self, name):
        return self.functions.get(name, None)



# Estruturas de AST
class ASTNode:
    pass

class BinOp(ASTNode):
    def __init__(self, op, left, right):
        self.op = op
        self.left = left
        self.right = right

class Var(ASTNode):
    def __init__(self, name):
        self.name = name

class Const(ASTNode):
    def __init__(self, value):
        self.value = value

class SemanticAnalyzer:
    def __init__(self):
        self.errors = []
        self.function_table = FunctionTable()

    def get_type(self, node, table):
        if node[0] == 'INTEGER':
            return 'int'
        elif node[0] == 'BOOLEAN':
            return 'bool'
        elif node[0] == 'STRING':
            return 'string'
        elif node[0] == 'IDENTIFIER':
            var_type = table.lookup(node[1])
            return var_type if var_type else 'undef'
        elif node[0] == 'BIN_OP':
            left_type = self.get_type(node[2], table)
            right_type = self.get_type(node[3], table)
            if left_type != right_type:
                self.errors.append(
                    f"Erro semântico: tipos incompatíveis em operação binária: '{left_type}' e '{right_type}'."
                )
                return 'undef'
            
            if node[1] in ('EQUALS', 'NOT_EQUALS', 'LESS_THAN', 'LESS_EQUAL', 'GREATER_THAN', 'GREATER_EQUAL'):
                    return 'bool'

            elif node[1] == 'PLUS' and left_type in ('int', 'string'):
                    return left_type
            elif node[1] in ('MINUS', 'MULTIPLY', 'DIVIDE') and left_type == 'int':
                    return 'int'
            elif node[1] in ('AND', 'OR') and left_type == 'bool':
                    return 'bool'
                
            if node[1] == 'PLUS' and left_type in ('int', 'string'):
                return left_type
            elif left_type in ('int', 'bool'):  # outras operações válidas
                return left_type
            else:
                self.errors.append(f"Erro semântico: operação inválida com tipo '{left_type}'.")
                return 'undef'
        elif node[0] == 'CALL':
            func_info = self.function_table.lookup(node[1])
            if not func_info:
                self.errors.append(f"Erro semântico: função '{node[1]}' não declarada.")
                return 'undef'

            ret_type, param_types = func_info
            args = node[2]

            if len(args) != len(param_types):
                self.errors.append(
                    f"Erro semântico: número incorreto de argumentos na chamada de '{node[1]}'. Esperado {len(param_types)}, recebido {len(args)}."
                )

            for idx, arg in enumerate(args):
                arg_type = self.get_type(arg, table)
                if idx < len(param_types) and arg_type != param_types[idx]:
                    self.errors.append(
                        f"Erro semântico: tipo incorreto no argumento {idx + 1} da função '{node[1]}'. Esperado '{param_types[idx]}', recebido '{arg_type}'."
                    )

            return ret_type
        
                        
        elif node[0] == 'UNARY_OP' and node[1] == 'NOT':
            expr_type = self.get_type(node[2], table)
            if expr_type != 'bool':
                self.errors.append(
                    f"Erro semântico: operador '!' aplicado a tipo inválido '{expr_type}'."
                )
                return 'undef'
            return 'bool'
        
        elif node[0] == 'TERNARY':
            cond_type = self.get_type(node[1], table)
            true_type = self.get_type(node[2], table)
            false_type = self.get_type(node[3], table)

            if cond_type != 'bool':
                self.errors.append("Erro semântico: condição do operador ternário precisa ser bool.")
                return 'undef'

            if true_type == 'undef' or false_type == 'undef':
                return 'undef'

            if true_type != false_type:
                self.errors.append("Erro semântico: os dois ramos do operador ternário devem ter o mesmo tipo.")
                return 'undef'

            return true_type
        
        elif node[0] == 'ARRAY_ACCESS':
            nome = node[1]
            index_type = self.get_type(node[2], table)
            if index_type != 'int':
                self.errors.append(f"Erro semântico: índice de vetor deve ser int, encontrado '{index_type}'.")
                return 'undef'

            var_type = table.lookup(nome)
            if not var_type or not var_type.startswith('vector'):
                self.errors.append(f"Erro semântico: '{nome}' não é um vetor declarado.")
                return 'undef'

            # Extrair tipo interno
            inner_type = var_type.split('(')[1].split(',')[0]
            return inner_type

        elif node[0] == 'MATRIX_ACCESS':
            nome = node[1]
            row_type = self.get_type(node[2], table)
            col_type = self.get_type(node[3], table)
            if row_type != 'int' or col_type != 'int':
                self.errors.append(f"Erro semântico: índices da matriz devem ser int.")
                return 'undef'

            var_type = table.lookup(nome)
            if not var_type or not var_type.startswith('matrix'):
                self.errors.append(f"Erro semântico: '{nome}' não é uma matriz declarada.")
                return 'undef'

            inner_type = var_type.split('(')[1].split(',')[0]
            return inner_type

        elif node[0] == 'ASSIGN':
            return self.get_type(node[2], table)
        return 'undef'

    def analyze(self, node, table=None):
        if table is None:
            table = SymbolTable()

        if isinstance(node, tuple):
            kind = node[0]

            if kind == 'PROGRAMA' or kind == 'BLOCK':
                new_table = SymbolTable(table)
                for subnode in node[1:]:
                    self.analyze(subnode, new_table)

            elif kind in ('DECL', 'DECL_ASSIGN'):
                tipo = node[1]
                nome = node[2]
                try:
                    table.define(nome, tipo)
                except ValueError as ve:
                    self.errors.append(str(ve))
                if kind == 'DECL_ASSIGN':
                    expr = node[3]
                    valor_tipo = self.get_type(expr, table)
                    if valor_tipo != tipo:
                        self.errors.append(
                            f"Erro semântico: atribuição de tipo '{valor_tipo}' à variável '{nome}' de tipo '{tipo}'."
                        )
                    self.analyze(expr, table)

            elif kind == 'EXPR':
                self.analyze(node[1], table)

            elif kind == 'ASSIGN':
                var_node = node[1]
                expr_node = node[2]
                if var_node[0] != 'IDENTIFIER':
                    self.errors.append("Erro semântico: lado esquerdo da atribuição deve ser uma variável.")
                else:
                    var_name = var_node[1]
                    expected_type = table.lookup(var_name)
                    actual_type = self.get_type(expr_node, table)
                    if expected_type:
                        if expected_type != actual_type and actual_type != 'undef':
                            self.errors.append(f"Erro semântico: atribuição de tipo '{actual_type}' à variável '{var_name}' de tipo '{expected_type}'.")
                    else:
                        self.errors.append(f"Erro semântico: variável '{var_name}' não declarada.")
                self.analyze(expr_node, table)


            elif kind == 'IF' or kind == 'WHILE':
                self.analyze(node[1], table)
                self.analyze(node[2], table)

            elif kind == 'RETURN':
                self.analyze(node[1], table)

            elif kind == 'BIN_OP':
                self.analyze(node[2], table)
                self.analyze(node[3], table)

            elif kind == 'IDENTIFIER':
                var_name = node[1]
                if not table.lookup(var_name):
                    self.errors.append(f"Erro semântico: variável '{var_name}' não declarada.")
                    
            elif kind == 'DECL_ASSIGN':
                tipo = node[1]
                nome = node[2]
                expr = node[3]
                try:
                    table.define(nome, tipo)
                except ValueError as ve:
                    self.errors.append(str(ve))
                valor_tipo = self.get_type(expr, table)
                if valor_tipo != tipo:
                    self.errors.append(
                        f"Erro semântico: atribuição de tipo '{valor_tipo}' à variável '{nome}' de tipo '{tipo}'."
                    )
                self.analyze(expr, table)
                
            elif node[0] == 'STRING':
                return 'string'
            
            elif kind == 'CALL':
                for arg in node[2]:
                    self.analyze(arg, table)
            elif kind == 'FUNCAO':
                tipo_retorno = node[1]
                nome_funcao = node[2]
                parametros = node[3]
                corpo = node[4]
                
                self.function_table.define(
                    nome_funcao,
                    tipo_retorno,
                    [param_tipo for param_tipo, _ in parametros]
                )

                # Criar escopo novo com os parâmetros
                func_table = SymbolTable(table)
                for param_tipo, param_nome in parametros:
                    try:
                        func_table.define(param_nome, param_tipo)
                    except ValueError as ve:
                        self.errors.append(str(ve))

                # Analisar o corpo da função no novo escopo
                self.analyze_func_body(corpo, func_table, tipo_retorno)

            elif kind == 'INTEGER':
                pass
            
    def analyze_func_body(self, node, table, expected_return_type):
        if isinstance(node, tuple):
            kind = node[0]

            if kind == 'RETURN':
                retorno_tipo = self.get_type(node[1], table)
                if retorno_tipo != expected_return_type:
                    self.errors.append(
                        f"Erro semântico: tipo de retorno '{retorno_tipo}' incompatível com tipo da função '{expected_return_type}'."
                    )
                self.analyze(node[1], table)

            elif kind == 'BLOCK':
                new_table = SymbolTable(table)
                for subnode in node[1:]:
                    self.analyze_func_body(subnode, new_table, expected_return_type)

            else:
                self.analyze(node, table)

# Funcoes GUI
def analyze_code(code):
    try:
        tokens = lexer(code)
        parser = Parser(tokens)
        ast = parser.parse_programa()

        semantic = SemanticAnalyzer()
        semantic.analyze(ast)
        global tac, temp_counter
        tac = []
        temp_counter = 0
        generate_TAC(ast)
        
        with open('output.tac', 'w') as f:
            for instr in tac:
                f.write(instr + '\n')
        
        sem_errors = "\n".join(semantic.errors)
        if sem_errors:
            return f"{print_ast(ast)}\n{sem_errors}\n\nTAC:\n" + "\n".join(tac), ast
        else:
            return f"{print_ast(ast)}\n\nTAC:\n" + "\n".join(tac), ast
    except (ValueError, SyntaxError) as e:
        return str(e), None

def show_ast():
    code = code_input.get("1.0", tk.END).strip()
    ast_output.delete("1.0", tk.END)
    text, _ = analyze_code(code)
    ast_output.insert(tk.END, text)

def visualize_ast():
    code = code_input.get("1.0", tk.END).strip()
    ast_output.delete("1.0", tk.END)
    text, ast = analyze_code(code)
    if ast is None:
        ast_output.insert(tk.END, text)
        return
    try:
        graph = build_graphviz_ast(ast)
        graph.render('ast', format='png', cleanup=True)
        img = Image.open('ast.png').resize((600, 400))
        img_tk = ImageTk.PhotoImage(img)
        img_label.configure(image=img_tk)
        img_label.image = img_tk
        ast_output.insert(tk.END, "Árvore sintática gerada com sucesso.")
    except Exception as e:
        ast_output.insert(tk.END, f"Erro ao gerar imagem: {str(e)}")

# Interface grafica
root = tk.Tk()
root.title("Analisador Léxico, Sintático e Semântico com AST Visual")

code_input = scrolledtext.ScrolledText(root, width=70, height=10)
code_input.pack(pady=10)

btn_frame = tk.Frame(root)
btn_frame.pack()

tk.Button(btn_frame, text="Analisar Texto", command=show_ast).pack(side=tk.LEFT, padx=5)
tk.Button(btn_frame, text="Visualizar Árvore", command=visualize_ast).pack(side=tk.LEFT, padx=5)

ast_output = scrolledtext.ScrolledText(root, width=70, height=7)
ast_output.pack(pady=10)

img_label = tk.Label(root)
img_label.pack(pady=10)

root.mainloop()