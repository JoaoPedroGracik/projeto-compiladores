import re
import tkinter as tk
from tkinter import scrolledtext
from graphviz import Digraph
from PIL import Image, ImageTk

# Tokens e expressoes regulares
TOKEN_REGEX = [
    (r'int\b', 'INT'),
    (r'if\b', 'IF'),
    (r'while\b', 'WHILE'),
    (r'return\b', 'RETURN'),
    (r'true\b|false\b', 'BOOLEAN'),
    (r'[a-zA-Z_][a-zA-Z0-9_]*', 'IDENTIFIER'),
    (r'\d+', 'INTEGER'),
    (r'==', 'EQUALS'),
    (r'=', 'ASSIGN'),
    (r'!=', 'NOT_EQUALS'),
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
    (r'\{', 'LBRACE'),
    (r'\}', 'RBRACE'),
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

    def parse_programa(self):
        return ('PROGRAMA', *self.parse_lista_de_comandos())

    def parse_lista_de_comandos(self):
        comandos = []
        while self.pos < len(self.tokens) and self.current_token()[0] != 'RBRACE':
            comandos.append(self.parse_comando())
        return comandos

    def parse_comando(self):
        token = self.current_token()
        if token[0] == 'INT':
            return self.parse_declaracao()
        elif token[0] == 'IF':
            return self.parse_if_comando()
        elif token[0] == 'WHILE':
            return self.parse_while_comando()
        elif token[0] == 'RETURN':
            return self.parse_return_comando()
        elif token[0] == 'LBRACE':
            return self.parse_bloco()
        else:
            return self.parse_expressao_comando()

    def parse_declaracao(self):
        self.eat('INT')
        var_name = self.current_token()[1]
        self.eat('IDENTIFIER')
        self.eat('SEMICOLON')
        return ('DECL', 'int', var_name)

    def parse_if_comando(self):
        self.eat('IF')
        self.eat('LPAREN')
        expr = self.parse_expressao()
        self.eat('RPAREN')
        comando = self.parse_comando()
        return ('IF', expr, comando)

    def parse_while_comando(self):
        self.eat('WHILE')
        self.eat('LPAREN')
        expr = self.parse_expressao()
        self.eat('RPAREN')
        comando = self.parse_comando()
        return ('WHILE', expr, comando)

    def parse_return_comando(self):
        self.eat('RETURN')
        expr = self.parse_expressao()
        self.eat('SEMICOLON')
        return ('RETURN', expr)

    def parse_expressao_comando(self):
        expr = self.parse_expressao()
        self.eat('SEMICOLON')
        return ('EXPR', expr)

    def parse_bloco(self):
        self.eat('LBRACE')
        comandos = self.parse_lista_de_comandos()
        self.eat('RBRACE')
        return ('BLOCK', *comandos)

    def parse_expressao(self):
        node = self.parse_relacional()
        if self.current_token()[0] == 'ASSIGN':
            self.eat('ASSIGN')
            right = self.parse_expressao()
            return ('ASSIGN', node, right)
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
        else:
            raise SyntaxError(f"Erro sintático: token inesperado {token[0]}")

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

class SemanticAnalyzer:
    def __init__(self):
        self.errors = []

        def get_type(self, node, table):
            if node[0] == 'INTEGER':
                return 'int'
            elif node[0] == 'BOOLEAN':
                return 'bool'
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
                return left_type
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

            elif kind == 'DECL':
                try:
                    table.define(node[2], node[1])
                except ValueError as ve:
                    self.errors.append(str(ve))

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

            elif kind == 'INTEGER':
                pass


# Funcoes GUI

def analyze_code(code):
    try:
        tokens = lexer(code)
        parser = Parser(tokens)
        ast = parser.parse_programa()

        semantic = SemanticAnalyzer()
        semantic.analyze(ast)
        sem_errors = "\n".join(semantic.errors)
        if sem_errors:
            return f"{print_ast(ast)}\n{sem_errors}", ast
        else:
            return print_ast(ast), ast
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